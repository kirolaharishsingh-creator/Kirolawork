# Hold a clip's exposure steady: per frame, fit out = a*in + b (per channel) so the background level and the
# chair's dark tone match the reference frame, smooth a/b over time, apply. Fixes a light drift inside a shot.
# Usage: python3 tools/lock_exposure.py in.mp4 out.mp4 [ref_frame=0 | prev_clip.mp4]  (prev clip: match its last frame)
import sys, subprocess, cv2, numpy as np
cap = cv2.VideoCapture(sys.argv[1]); fps = cap.get(cv2.CAP_PROP_FPS); frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
ref = sys.argv[3] if len(sys.argv) > 3 else '0'
def anchors(f):
    f = f.astype(np.float32); l = f.mean(2)
    return f[l > 150].mean(0), np.median(f[l < 110], 0)
if ref.endswith('.mp4'):
    rc = cv2.VideoCapture(ref); rc.set(cv2.CAP_PROP_POS_FRAMES, rc.get(cv2.CAP_PROP_FRAME_COUNT) - 1); tb, tc = anchors(rc.read()[1])
else:
    tb, tc = anchors(frames[int(ref)])
A, B = [], []
for f in frames:
    bg, ch = anchors(f); a = (tb - tc) / np.maximum(bg - ch, 1); A.append(a); B.append(tb - a * bg)
k = cv2.getGaussianKernel(15, 4).ravel()
sm = lambda x: np.stack([np.convolve(np.pad(x[:, c], 7, mode='edge'), k, 'valid') for c in range(3)], 1)
A, B = sm(np.array(A)), sm(np.array(B))
if not ref.endswith('.mp4'): A, B = A - A[0] + 1, B - B[0]          # reference frame stays untouched
H, W = frames[0].shape[:2]
def encode(out, frs):
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(fps), '-i', '-',
                          '-vf', 'scale=flags=accurate_rnd+full_chroma_int:out_color_matrix=bt601:out_range=tv',
                          '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
    for f in frs: p.stdin.write(f.tobytes())
    p.stdin.close(); p.wait()
# the BGR -> yuv420p -> BGR round trip shifts levels slightly; measure it once and pre-compensate
encode(sys.argv[2], [frames[0]] * 3)
bias = frames[0].astype(np.float32).mean((0, 1)) - cv2.VideoCapture(sys.argv[2]).read()[1].astype(np.float32).mean((0, 1))
encode(sys.argv[2], (np.clip(np.rint(f.astype(np.float32) * a + b + bias), 0, 255).astype(np.uint8) for f, a, b in zip(frames, A, B)))
print('gain first/last', A[0].round(3), A[-1].round(3), 'offset last', B[-1].round(1))
