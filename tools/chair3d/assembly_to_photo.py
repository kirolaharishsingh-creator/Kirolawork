# Assembly video: 3D parts fly in and build the chair (anim3.py ASSEMBLE=1 frames), then the finished 3D chair dissolves
# into the real product photo, which holds to the end. The render frames are fitted onto the photo with the affine W from
# an ECC fit of the two silhouettes (render coords -> photo coords), so the photo stays at its native sharpness.
# Usage: python3 assembly_to_photo.py frames_dir photo.png out.mp4
#   env: W_NPY (saved 2x3 fit; fitted on the last render frame if missing), D0 first dissolve frame, D1 last dissolve
#        frame (default: last rendered frame), TOTAL total frames (216 = 9 s at 24 fps), FPS (24)
import os, sys, glob, subprocess, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comp import studio_bg, fill_pinholes

fdir, photo_path, out = sys.argv[1:4]
files = sorted(glob.glob(os.path.join(fdir, '*.png')))
photo = cv2.imread(photo_path); H, W = photo.shape[:2]
bg = (studio_bg(W, H)[..., ::-1] * 255).astype(np.float32)
TOTAL, FPS = int(os.environ.get('TOTAL', 216)), int(os.environ.get('FPS', 24))
D1 = int(os.environ.get('D1', len(files) - 1)); D0 = int(os.environ.get('D0', D1 - 15))

def fit(rgba):            # silhouettes of the dark chair in render and photo -> affine render->photo
    mr = ((rgba[..., 3] > 200) & (rgba[..., :3].mean(2) < 110)).astype(np.float32)
    mp = (photo.mean(2) < 110).astype(np.float32)
    A = np.eye(2, 3, dtype=np.float32)
    _, A = cv2.findTransformECC(cv2.GaussianBlur(mr, (0, 0), 3), cv2.GaussianBlur(mp, (0, 0), 3), A, cv2.MOTION_AFFINE,
                                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6), None, 5)
    return A                # maps render coords -> photo coords (photo(A x) ~ render(x))

wpath = os.environ.get('W_NPY', '')
A = np.load(wpath) if wpath and os.path.exists(wpath) else fit(cv2.imread(files[-1], cv2.IMREAD_UNCHANGED))
print('fit', np.round(A, 4).tolist())

def clean(r):              # model flaws: see-through pin-holes, thin dark spikes on rims, bright texture specks
    r = cv2.cvtColor(fill_pinholes(cv2.cvtColor(r, cv2.COLOR_BGRA2RGBA)), cv2.COLOR_RGBA2BGRA)
    solid = ((r[..., 3] > 128) & (r[..., :3].mean(2) < 90)).astype(np.uint8)
    k = int(os.environ.get('SPIKE', 5))
    opened = cv2.morphologyEx(solid, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    spike = (solid > 0) & (opened == 0)
    near = cv2.dilate(opened, np.ones((3, 3), np.uint8)) > 0                         # rim pixels next to a solid part
    spike &= cv2.dilate((r[..., 3] < 60).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0   # only where it sticks out
    upper = np.zeros(spike.shape, bool); upper[:int(os.environ.get('CLEAN_YMAX', 560))] = True   # back/headrest/lumbar only
    spike &= upper
    r = r.copy(); kill = (cv2.dilate((spike & ~near).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~(opened > 0) & upper
    r[kill, 3] = 0
    lum = r[..., :3].mean(2).astype(np.float32); med = cv2.medianBlur(r[..., :3], 9)
    speck = (solid > 0) | (cv2.dilate(solid, np.ones((9, 9), np.uint8)) > 0) & (r[..., 3] > 128)
    speck &= (lum - med.mean(2) > float(os.environ.get("SPECK_T", 22))) & (med.mean(2) < 60) & upper         # bright dot in a black part (not chrome)
    n, lab, st, _ = cv2.connectedComponentsWithStats(speck.astype(np.uint8))
    speck = np.isin(lab, [j for j in range(1, n) if st[j, 4] <= 40])  # dots only
    speck = (cv2.dilate(speck.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & upper
    med2 = cv2.medianBlur(r[..., :3], 21); r[speck, :3] = med2[speck]
    return r

def render_frame(i):      # render RGBA warped into photo coords, over the studio backdrop
    r = clean(cv2.imread(files[min(i, len(files) - 1)], cv2.IMREAD_UNCHANGED)).astype(np.float32)   # chair is still after the last render
    r = cv2.warpAffine(r, A, (W, H), flags=cv2.INTER_LANCZOS4, borderValue=(0, 0, 0, 0))   # render pixel x -> A x
    a = np.clip(r[..., 3:] / 255, 0, 1)
    return r[..., :3] * a + bg * (1 - a)

p = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS),
                      '-i', '-', '-an', '-c:v', 'libx264', '-crf', '14', '-preset', 'slow', '-pix_fmt', 'yuv420p',
                      '-movflags', '+faststart', out], stdin=subprocess.PIPE)
ph = photo.astype(np.float32)
# the photo's flat backdrop shows 8-bit banding rings: keep the chair and its floor shadow, take the backdrop from studio_bg
dev = cv2.GaussianBlur(np.abs(ph - bg).max(2), (0, 0), 1.5)
keep = np.clip((dev - 3.0) / 7.0, 0, 1); keep = (keep * keep * (3 - 2 * keep))[..., None]   # rings (1-4 levels) -> backdrop
ph = ph * keep + bg * (1 - keep)
rng = np.random.default_rng(0)
def dither(f): return f + rng.uniform(-0.6, 0.6, f.shape[:2])[..., None]   # breaks up gradient banding in the encode
# REVEAL=sweep (default): a soft diagonal light band crosses the chair; behind it the real photo, ahead of it the 3D
# render, so small shape differences between model and product are hidden by the moving light. REVEAL=dissolve: crossfade.
REVEAL = os.environ.get('REVEAL', 'sweep')
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
chair_ph = (ph.mean(2) < 110)
ys_, xs_ = np.where(chair_ph); X0, X1 = xs_.min() - 120, xs_.max() + 120
coord = xx + 0.30 * (yy - H / 2)                  # leaning band
BAND = float(os.environ.get('BAND', 70))
for i in range(TOTAL):
    if i < D0: f = render_frame(i)
    elif i <= D1:
        t = (i - D0) / max(D1 - D0, 1); t = t * t * (3 - 2 * t)
        rf = render_frame(i)
        if REVEAL == 'dissolve': f = rf * (1 - t) + ph * t
        else:
            pos = X0 + (X1 - X0) * t
            m = np.clip((pos - coord) / BAND + 0.5, 0, 1)[..., None]          # 1 = photo side
            m = m * m * (3 - 2 * m)
            f = rf * (1 - m) + ph * m
            dark = cv2.GaussianBlur(((f.mean(2) < 120)).astype(np.float32), (0, 0), 3)[..., None]
            glow = np.exp(-((coord - pos) / (BAND * 0.45)) ** 2)[..., None] * np.sin(np.pi * t)
            f = f + glow * (dark * float(os.environ.get('GLOW_CHAIR', 55)) + (1 - dark) * float(os.environ.get('GLOW_WALL', 6)))                        # light catches the chair, faint on the wall
    else: f = ph
    p.stdin.write(np.clip(dither(f), 0, 255).astype(np.uint8).tobytes())
p.stdin.close(); p.wait(); print('ok', out)
