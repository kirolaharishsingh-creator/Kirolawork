# Armrest side-slide loop from a start/end frame pair (no AI): both real pads are lifted (soft alpha, un-mixed colour),
# the start frame's pad is removed (clean plate), then per frame the start pad moves +p*D and the end pad moves
# (p-1)*D while they cross-fade (weight p) - both sit at the same spot at every instant, so the frame starts on the
# exact start pad and ends on the exact end pad. D = measured pad offset (phase correlation). Smoothstep easing.
# Usage: python armside_pair_slide.py start.png end.png out.mp4 [out_small.mp4]
#   env: SEAM 521-style seam at 1920 (551), KEYS t:p (default hold/out/hold/back/hold, 4 s), FPS 30
import cv2, numpy as np, sys, os, subprocess, runpy, tempfile
s_p, e_p, out = sys.argv[1:4]; small = sys.argv[4] if len(sys.argv) > 4 else None
os.environ.setdefault('SEAM', '551')
tmp = tempfile.mktemp(suffix='.png'); sys.argv = ['x', s_p, e_p, tmp]    # reuse the lift() of armside_pad_swap.py
g = runpy.run_path(os.path.join(os.path.dirname(__file__), 'armside_pad_swap.py'), run_name='lib'); os.remove(tmp)
S, E, lift, gray = g['S'], g['E'], g['lift'], g['gray']; h, w = S.shape[:2]
clean, aS, FS = lift(S, gray(S)); _, aE, FE = lift(E, gray(E))
(dx, dy), r = cv2.phaseCorrelate(aS.astype(np.float32), aE.astype(np.float32))
print('pad offset start->end: %.2f, %.2f px (resp %.2f)' % (dx, dy, r))
FPS = int(os.environ.get('FPS', 30))
KEYS = [tuple(float(v) for v in k.split(':')) for k in os.environ.get('KEYS', '0:0,0.6:0,1.6:1,2.4:1,3.4:0,4:0').split(',')]
def at(t):
    for (t0, v0), (t1, v1) in zip(KEYS, KEYS[1:]):
        if t <= t1: u = (t - t0) / max(t1 - t0, 1e-6); u = u * u * (3 - 2 * u); return v0 + (v1 - v0) * u
    return KEYS[-1][1]
def sh(img, x, y): return cv2.warpAffine(img, np.float32([[1, 0, x], [0, 1, y]]), (w, h), flags=cv2.INTER_CUBIC)
encs = []
for path, size in ((out, (w, h)), (small, (1920, 1080))):
    if path: encs.append((size, subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '%dx%d' % size,
        '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-crf', '14', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path], stdin=subprocess.PIPE)))
rng = np.random.default_rng(3); n = int(round(KEYS[-1][0] * FPS))
for i in range(n):
    p = at(i / FPS)
    if p < 1e-3: fr = S.copy()
    elif p > 1 - 1e-3: fr = FE * aE[..., None] + clean * (1 - aE[..., None])
    else:
        a1, F1 = sh(aS, p * dx, p * dy), sh(FS * aS[..., None], p * dx, p * dy)
        a2, F2 = sh(aE, (p - 1) * dx, (p - 1) * dy), sh(FE * aE[..., None], (p - 1) * dx, (p - 1) * dy)
        am = (1 - p) * a1 + p * a2; P = (1 - p) * F1 + p * F2                # premultiplied cross-fade of colour,
        a = np.maximum(a1, a2); col = P / np.maximum(am, 1e-3)[..., None]     # union of the outlines: no see-through edge
        fr = col * a[..., None] + clean * (1 - a[..., None])
    fr = fr + rng.normal(0, 0.8, fr.shape)                                    # light live grain
    for size, ff in encs:
        im = np.clip(fr, 0, 255).astype(np.uint8)
        if size != (w, h): im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
        ff.stdin.write(im.tobytes())
for _, ff in encs: ff.stdin.close(); ff.wait()
print('frames', n, 'fps', FPS, 'size', w, h)
