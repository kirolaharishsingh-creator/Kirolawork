# Per-frame white balance + exposure: scale each channel so the studio background matches a target colour,
# smoothed over time. Gain only (no offset), so the chair's blacks and contrast are kept.
# Usage: python3 tools/match_bg.py in.mp4 out.mp4 B G R
import sys, subprocess, cv2, numpy as np
cap = cv2.VideoCapture(sys.argv[1]); fps = cap.get(cv2.CAP_PROP_FPS); frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
target = np.array([float(v) for v in sys.argv[3:6]], np.float32)
G = []
for f in frames:
    f = f.astype(np.float32); l = f.mean(2); G.append(target / f[l > 150].astype(np.float64).mean(0))
k = cv2.getGaussianKernel(15, 4).ravel(); G = np.array(G)
G = np.stack([np.convolve(np.pad(G[:, c], 7, mode='edge'), k, 'valid') for c in range(3)], 1)
H, W = frames[0].shape[:2]
def encode(out, frs):
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(fps), '-i', '-',
                          '-vf', 'scale=flags=accurate_rnd+full_chroma_int:out_color_matrix=bt601:out_range=tv',
                          '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
    for f in frs: p.stdin.write(f.tobytes())
    p.stdin.close(); p.wait()
encode(sys.argv[2], [frames[0]] * 3)               # measure the yuv round-trip level shift and pre-compensate
bias = frames[0].astype(np.float64).mean((0, 1)) - cv2.VideoCapture(sys.argv[2]).read()[1].astype(np.float64).mean((0, 1))
encode(sys.argv[2], (np.clip(np.rint(f.astype(np.float32) * g + bias), 0, 255).astype(np.uint8) for f, g in zip(frames, G)))
print('gain first/last', G[0].round(3), G[-1].round(3))
