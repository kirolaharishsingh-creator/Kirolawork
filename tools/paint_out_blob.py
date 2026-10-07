# Remove a small bright artefact (e.g. an invented chrome plate) from a clip: track the bright blob from a seed
# point, frame by frame, and inpaint it with the surrounding (dark) colour. Frames where it isn't found stay untouched.
# Usage: python3 tools/paint_out_blob.py in.mp4 out.mp4 seed_frame x y first_frame last_frame [thr=150] [max_area=6000]
import sys, subprocess, cv2, numpy as np
inp, out, sf, sx, sy = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
F0, F1 = int(sys.argv[6]), int(sys.argv[7])          # only touch frames in this range (keeps real details outside it)
THR = float(sys.argv[8]) if len(sys.argv) > 8 else 150
MAX_AREA = int(sys.argv[9]) if len(sys.argv) > 9 else 8000
R = 75                                                 # pieces of the blob within this radius of the track belong to it
cap = cv2.VideoCapture(inp); fps = cap.get(cv2.CAP_PROP_FPS); frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
H, W = frames[0].shape[:2]

def find(f, px, py, prev_area):
    l = f.astype(np.float64).mean(2)
    n, lab, st, cen = cv2.connectedComponentsWithStats((l > THR).astype(np.uint8))
    keep = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 40 or a > MAX_AREA or x == 0 or y == 0 or x + w >= W or y + h >= H: continue   # not the open background
        if np.hypot(cen[i][0] - px, cen[i][1] - py) < R: keep.append(i)
    if not keep: return None
    m = np.isin(lab, keep); ys, xs = np.nonzero(m)
    return m, (xs.mean(), ys.mean()), len(xs)

masks = {}
for direction in (1, -1):
    px, py, area = sx, sy, None
    i = sf
    while F0 <= i <= F1:
        r = find(frames[i], px, py, area)
        if r is None: break
        m, (px, py), area = r
        masks[i] = m
        i += direction
print('blob found in frames', min(masks), '-', max(masks), f'({len(masks)} frames)')

def encode(path, frs):
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(fps), '-i', '-',
                          '-vf', 'scale=flags=accurate_rnd+full_chroma_int:out_color_matrix=bt601:out_range=tv',
                          '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', path], stdin=subprocess.PIPE)
    for f in frs: p.stdin.write(f.tobytes())
    p.stdin.close(); p.wait()
encode(out, [frames[0]] * 3)                      # measure the yuv round-trip level shift and pre-compensate
bias = frames[0].astype(np.float64).mean((0, 1)) - cv2.VideoCapture(out).read()[1].astype(np.float64).mean((0, 1))
GROW = 100      # the plate's grey rim is darker than its bright core: grow the core into pixels above this level
def full_mask(f, core):
    l = f.astype(np.float64).mean(2)
    ys, xs = np.nonzero(core); cx, cy = xs.mean(), ys.mean()
    r = max(np.ptp(xs), np.ptp(ys)) * 0.9 + 12
    yy, xx = np.mgrid[0:H, 0:W]
    near = (xx - cx) ** 2 + (yy - cy) ** 2 < r * r
    n, lab, st, _ = cv2.connectedComponentsWithStats(((l > GROW) & near).astype(np.uint8))
    m = np.zeros((H, W), bool)
    for k in range(1, n):
        if (core & (lab == k)).any(): m |= lab == k            # grey rim attached to the bright core
    bg = l > 175                                               # never paint the open studio background
    nb, lb, sb, _ = cv2.connectedComponentsWithStats(bg.astype(np.uint8))
    big = np.isin(lb, [k for k in range(1, nb) if sb[k][4] > 20000])
    return m & ~big | core
def fixed():
    for i, f in enumerate(frames):
        if i in masks:
            m = cv2.dilate(full_mask(f, masks[i]).astype(np.uint8), np.ones((11, 11), np.uint8))
            ring = cv2.dilate(m, np.ones((31, 31), np.uint8)).astype(bool) & ~m.astype(bool)
            l = f.astype(np.float64).mean(2)
            dark = ring & (l < 70)                                   # the black plastic spine around the plate
            col = np.median(f[dark], 0) if dark.sum() > 50 else np.median(f[ring], 0)
            fill = cv2.inpaint(f, m * 255, 12, cv2.INPAINT_TELEA).astype(np.float64)
            fill = 0.25 * fill + 0.75 * col                          # mostly the spine colour, a little local shading
            a = cv2.GaussianBlur(m.astype(np.float64), (0, 0), 3)[..., None]
            f = (f * (1 - a) + fill * a).astype(np.uint8)
        yield np.clip(np.rint(f.astype(np.float64) + bias), 0, 255).astype(np.uint8)
encode(out, fixed())
