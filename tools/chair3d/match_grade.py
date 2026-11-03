# Colour grade that MATCHES a reference video's look (client: "the recline video's grade is perfect, the others look
# darker"). Pure grade, no compositing or retiming: one fixed curve per clip (no flicker).
#  - measures both clips over sampled frames: chair tones (dark pixels: 5/25/50/75th percentiles) and the wall
#    (light pixels: median, mean colour)
#  - per-channel gains make the wall the reference's wall colour; then one smooth monotone tone curve (PCHIP through the
#    matched percentiles) maps the clip's chair tones onto the reference's, so shadows keep the same lifted detail
# Usage: python match_grade.py reference.mp4 in.mp4 out.mp4   (env: SAT 1.0, CRF 14)
import os, sys, subprocess, numpy as np, cv2
from scipy.interpolate import PchipInterpolator
ref, src, out = sys.argv[1:4]
SAT = float(os.environ.get('SAT', 1.0))
def stats(path, gains=None):
    cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fr = []
    for i in np.linspace(0, n - 1, 12).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = cap.read()
        if ok: fr.append(cv2.resize(f, (960, int(960 * f.shape[0] / f.shape[1])), interpolation=cv2.INTER_AREA))
    A = np.concatenate([f.reshape(-1, 3) for f in fr]).astype(np.float32)
    if gains is not None: A = np.clip(A * gains, 0, 255)
    L = A @ np.array([0.114, 0.587, 0.299], np.float32)                 # BGR luma
    d, wl = L[L < 110], L > 140
    return np.percentile(d, [5, 25, 50, 75]), np.median(L[wl]), A[wl].mean(0)
rp, rw, rcol = stats(ref)
_, _, scol = stats(src)
g = (rcol / scol) * (scol.mean() / rcol.mean())                          # wall colour balance only (keep its level)
sp, sw, _ = stats(src, g)
xs = np.r_[0, sp, sw, 255.]; ys = np.r_[0, rp, rw, 255.]
keep = np.r_[True, np.diff(xs) > 1.5]; xs, ys = xs[keep], np.maximum.accumulate(ys[keep])
curve = PchipInterpolator(xs, ys)(np.arange(256))
print('ref chair p5/25/50/75', rp.round(1), 'wall', round(rw, 1), '| src (balanced)', sp.round(1), round(sw, 1))
lut = np.stack([np.clip(curve[np.clip(np.arange(256) * g[c], 0, 255).astype(int)], 0, 255) for c in range(3)], 1).astype(np.uint8)[:, None, :]
cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS)
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
p = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', f'{fps}', '-i', '-',
                      '-c:v', 'libx264', '-crf', os.environ.get('CRF', '14'), '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out],
                     stdin=subprocess.PIPE)
k = 0
while True:
    ok, f = cap.read()
    if not ok: break
    f = cv2.LUT(f, lut)
    if SAT != 1:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32); hsv[..., 1] *= SAT
        f = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    p.stdin.write(f.tobytes()); k += 1
p.stdin.close(); p.wait(); print('ok', out, k, 'frames @', fps)
