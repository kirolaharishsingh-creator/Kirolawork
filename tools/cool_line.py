# Cool a warm highlight line towards the cool white used in the other shots (background untouched).
# Usage: python3 tools/cool_line.py in.mp4 out.mp4
import cv2, numpy as np, subprocess, sys
cap = cv2.VideoCapture(sys.argv[1]); fps = cap.get(cv2.CAP_PROP_FPS)
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(fps), '-i', '-',
                      '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', sys.argv[2]], stdin=subprocess.PIPE)
while True:
    ok, f = cap.read()
    if not ok: break
    f = f.astype(np.float32); lum = f.mean(2)
    onchair = cv2.blur((lum < 60).astype(np.float32), (31, 31))
    w = (np.clip((lum - 120) / 60, 0, 1) * np.clip((onchair - 0.6) / 0.15, 0, 1))[..., None]
    p.stdin.write(np.clip(f * (1 + w * np.array([0.12, 0.03, -0.08])), 0, 255).astype(np.uint8).tobytes())
p.stdin.close(); p.wait()
