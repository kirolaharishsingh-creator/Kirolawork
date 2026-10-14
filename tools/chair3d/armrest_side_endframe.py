# Armrest 4th movement end frame from ONE photo: the pad is lifted out (soft alpha from its darkness against the
# backdrop), the hole is refilled (inpainted wall, post continued up under the pad), and the pad is put back DX px
# sideways, outward. Pixel-identical everywhere else, so Kling sees no camera drift.
# Writes start/end at the photo's size and upscaled to OUT_W (16:9, Lanczos + light unsharp, identical for both).
# Usage: python armrest_side_endframe.py photo.jpg outdir
#   env: BOX x0,y0,x1,y1 (pad box, bottom = pad/post seam)  SEED x,y (a pad pixel)  DX 32  OUT_W 1920,3840
import cv2, numpy as np, sys, os
src, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
I = cv2.imread(src).astype(np.float32); h, w = I.shape[:2]
x0, y0, x1, y1 = [int(v) for v in os.environ.get('BOX', '404,288,560,424').split(',')]
sx, sy = [int(v) for v in os.environ.get('SEED', '475,360').split(',')]
DX = int(os.environ.get('DX', 32))
L = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)

row = L[y1 + 4]; cols = np.where(row[x0:x1] < 90)[0] + x0                       # post run just below the seam
px0, px1 = cols.min(), cols.max(); print('post cols', px0, px1)
# hard pad mask: dark pixels in the box (+8 px below the seam beside the post: the pad's lower rim), seed component
dark = np.zeros((h, w), np.uint8); dark[y0:y1 + 8, x0:x1] = (L[y0:y1 + 8, x0:x1] < 110).astype(np.uint8)
dark[y1:, px0 - 2:px1 + 3] = 0
n, lab = cv2.connectedComponents(dark); pad = (lab == lab[sy, sx]).astype(np.uint8)
pad = cv2.morphologyEx(pad, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
print('pad px', int(pad.sum()), 'bbox', cv2.boundingRect(pad))
grow = cv2.dilate(pad, np.ones((7, 7), np.uint8)); grow[y1:, px0 - 3:px1 + 4] = 0   # never touch the post below the seam

# clean plate: wall inpainted, then the post continued straight up under where the pad was
clean = cv2.inpaint(I.astype(np.uint8), grow * 255, 9, cv2.INPAINT_TELEA).astype(np.float32)
ref = I[y1 + 2:y1 + 6, px0 - 2:px1 + 3].mean(0)
for y in range(y1 - 25, y1 + 2):                                             # only under the pad
    clean[y, px0 - 2:px1 + 3] = np.where(grow[y, px0 - 2:px1 + 3, None] > 0, ref, clean[y, px0 - 2:px1 + 3])

# soft alpha (edge anti-aliasing kept) and un-mixed pad colour
padL = np.median(L[pad > 0]); bgL = cv2.cvtColor(clean.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
a = np.clip((bgL - L) / np.maximum(bgL - padL, 1), 0, 1) * grow; a[pad > 0] = np.maximum(a[pad > 0], 0.0)
a = np.where(cv2.erode(pad, np.ones((3, 3), np.uint8)) > 0, 1.0, a)
F = (I - (1 - a[..., None]) * clean) / np.maximum(a[..., None], 0.05)
M = np.float32([[1, 0, DX], [0, 1, 0]])
a2 = cv2.warpAffine(a, M, (w, h)); F2 = cv2.warpAffine(F, M, (w, h))
E = F2 * a2[..., None] + clean * (1 - a2[..., None])
# seam between the moved pad's rim and the post: a 5x5 median removes 1-2 px leftovers of the old rim edge
bx0, bx1 = x0 + DX - 10, x1 + DX + 6; band = np.clip(E[y1 - 8:y1 + 10, bx0:bx1], 0, 255).astype(np.uint8)
E[y1 - 8:y1 + 10, bx0:bx1] = cv2.medianBlur(band, 5)
S = I
for ow in [int(v) for v in os.environ.get('OUT_W', '1920,3840').split(',')]:
    oh = ow * h // w if abs(w / h - 16 / 9) > 0.01 else ow * 9 // 16
    for nm, im in (('start', S), ('end', E)):
        up = cv2.resize(np.clip(im, 0, 255).astype(np.uint8), (ow, oh), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
        up = np.clip(up + 0.5 * (up - cv2.GaussianBlur(up, (0, 0), 1.2 * ow / 1920)), 0, 255).astype(np.uint8)
        cv2.imwrite('%s/armside_photo_%s_%d.png' % (out, nm, ow), up)
cv2.imwrite(out + '/armside_photo_end_native.png', np.clip(E, 0, 255).astype(np.uint8))
print('done', w, h, 'DX', DX)

# VIDEO=out.mp4: no AI, the slide itself rendered from the cut-out pad (sub-pixel shift, smoothstep easing),
# loop: hold, out, hold, back, hold (KEYS t:fraction-of-DX), 1080p, light per-frame grain so it doesn't read as a still
if os.environ.get('VIDEO'):
    import subprocess
    FPS = 30; KEYS = [tuple(float(v) for v in k.split(':')) for k in os.environ.get('KEYS', '0:0,0.6:0,1.6:1,2.4:1,3.4:0,4:0').split(',')]
    def at(t):
        for (t0, v0), (t1, v1) in zip(KEYS, KEYS[1:]):
            if t <= t1: u = (t - t0) / max(t1 - t0, 1e-6); u = u * u * (3 - 2 * u); return v0 + (v1 - v0) * u
        return KEYS[-1][1]
    ow, oh = 1920, 1080; rng = np.random.default_rng(7)
    ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '%dx%d' % (ow, oh), '-r', str(FPS), '-i', '-',
                           '-c:v', 'libx264', '-crf', '14', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', os.environ['VIDEO']], stdin=subprocess.PIPE)
    n = int(round(KEYS[-1][0] * FPS))
    for i in range(n):
        d = DX * at(i / FPS); Mi = np.float32([[1, 0, d], [0, 1, 0]])
        ai = cv2.warpAffine(a, Mi, (w, h), flags=cv2.INTER_CUBIC); Fi = cv2.warpAffine(F, Mi, (w, h), flags=cv2.INTER_CUBIC)
        fr = Fi * ai[..., None] + clean * (1 - ai[..., None]) if d > 0.01 else I.copy()
        if d > 0.01:
            band = np.clip(fr[y1 - 8:y1 + 10, x0 - 10:x1 + DX + 6], 0, 255).astype(np.uint8)
            fr[y1 - 8:y1 + 10, x0 - 10:x1 + DX + 6] = cv2.medianBlur(band, 5)
        up = cv2.resize(np.clip(fr, 0, 255).astype(np.uint8), (ow, oh), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
        up = up + 0.5 * (up - cv2.GaussianBlur(up, (0, 0), 1.2)) + rng.normal(0, 1.0, up.shape)
        ff.stdin.write(np.clip(up, 0, 255).astype(np.uint8).tobytes())
    ff.stdin.close(); ff.wait(); print('video', os.environ['VIDEO'], n, 'frames')
