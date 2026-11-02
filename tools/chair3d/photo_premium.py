# Premium light look for photo / Kling USP clips (same colour setting as premium_look.py's 3D renders):
#  - chair: premium_look.grade_chair (clarity, contrast with deeper blacks, chrome lift, cool neutral)
#  - wall: the warm beige studio wall is swapped for premium_look.backdrop('light'); the photo's own shading relative to
#    the smooth wall (floor contact shadows, soft falloff) is kept as a multiplier
#  - matte from brightness (chair dark, wall light), coloured pixels (e.g. the blue air lines of the mesh clip) are kept
#  - LOOP=pingpong: play forward then backward (seamless for one-way clips); LOOP=as_is: keep the clip's own loop
# Usage: python photo_premium.py in.mp4 out.mp4   (env: LOOP as_is|pingpong, OUT_W 1920, TRIM s0:s1)
import os, sys, subprocess, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from premium_look import backdrop, grade_chair
src, out = sys.argv[1:3]; LOOP = os.environ.get('LOOP', 'as_is'); OW = int(os.environ.get('OUT_W', 1920))
cap = cv2.VideoCapture(src); fps = cap.get(cv2.CAP_PROP_FPS); frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
if os.environ.get('TRIM'):
    a, b = [float(v) for v in os.environ['TRIM'].split(':')]; frames = frames[int(a * fps):int(b * fps)]
h0, w0 = frames[0].shape[:2]; OH = OW * h0 // w0
BG = backdrop(OW, OH, 'light')[..., ::-1] + np.random.default_rng(9).normal(0, 0.7, (OH, OW, 3))                                            # BGR
def grade(f):
    f = cv2.resize(f, (OW, OH), interpolation=cv2.INTER_LANCZOS4 if OW > w0 else cv2.INTER_AREA).astype(np.float32)
    L = cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    m = np.clip((L - 95) / 60, 0, 1); m = m * m * (3 - 2 * m)                          # 0 chair .. 1 wall
    blue = ((hsv[..., 0] > 85) & (hsv[..., 0] < 135) & (hsv[..., 2] > 120)).astype(np.float32) * np.clip((hsv[..., 1] - 10) / 20, 0, 1)
    blue = cv2.dilate(blue, np.ones((3, 3), np.uint8))                                  # the blue air lines: keep as shot
    wallw = (m > 0.95).astype(np.float32); sg = 40 * OW / 1920
    W0 = cv2.GaussianBlur(L * wallw, (0, 0), sg) / np.maximum(cv2.GaussianBlur(wallw, (0, 0), sg), 1e-3)
    W0 = np.where(W0 < 60, np.percentile(L[wallw > 0], 60), W0)
    s3 = 3 * OW / 1920; wl = (m > 0.6).astype(np.float32)                                # wall-only local level:
    Ls = cv2.GaussianBlur(L * wl, (0, 0), s3) / np.maximum(cv2.GaussianBlur(wl, (0, 0), s3), 1e-3)   # no chair leak
    Ls = np.where(cv2.GaussianBlur(wl, (0, 0), s3) < 0.05, W0, Ls)
    dip = np.clip((W0 - Ls) / np.maximum(W0, 1) - 0.035, 0, 1)                         # only real shadows (> 3.5 %)
    shade = (1 - dip / 0.965)[..., None]                                                # (compression banding ignored)
    bg = BG * shade
    chair = grade_chair(cv2.bilateralFilter(f.astype(np.uint8), 5, 12, 5)[..., ::-1], float(os.environ.get('K', 0.4)))[..., ::-1]                                       # grade_chair works in RGB
    out = chair * (1 - m[..., None]) + bg * m[..., None]
    out = out * (1 - blue[..., None]) + f * blue[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)
if LOOP == 'pingpong':
    # forward then backward with the speed eased to zero at both turnarounds (clips that start already moving),
    # in-between frames by blending neighbours
    n = len(frames); nout = int(n * 1.25)
    def at(t):
        i = int(np.floor(t)); a = t - i; j = min(i + 1, n - 1)
        return frames[i] if a < 1e-3 else cv2.addWeighted(frames[i], 1 - a, frames[j], a, 0)
    u = np.linspace(0, 1, nout); e = 0.35 * u + 0.65 * (u * u * (3 - 2 * u))
    fwd = [at(t) for t in e * (n - 1)]; seq = fwd + fwd[-2:0:-1]
else:
    seq = frames
p = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{OW}x{OH}', '-r', f'{fps}', '-i', '-',
                      '-an', '-c:v', 'libx264', '-crf', '14', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out],
                     stdin=subprocess.PIPE)
for f in seq: p.stdin.write(grade(f).tobytes())
p.stdin.close(); p.wait(); print('ok', out, len(seq), 'frames @', fps, 'loop', LOOP)
