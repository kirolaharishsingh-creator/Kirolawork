# Add a slow digital push-in (dolly-in look) to a clip: scale 1.0 -> END_SCALE over the clip, eased.
# Usage: python3 tools/video_push_in.py in.mp4 out.mp4 END_SCALE [cx cy]
import sys, subprocess, cv2, numpy as np
cap = cv2.VideoCapture(sys.argv[1]); fps = cap.get(cv2.CAP_PROP_FPS)
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); end = float(sys.argv[3])
cx = float(sys.argv[4]) if len(sys.argv) > 4 else W / 2; cy = float(sys.argv[5]) if len(sys.argv) > 5 else H / 2
p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(fps), '-i', '-',
                      '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', sys.argv[2]], stdin=subprocess.PIPE)
for i in range(n):
    ok, f = cap.read()
    if not ok: break
    t = i / max(n - 1, 1); s = 1 + (end - 1) * (t * t * (3 - 2 * t))
    M = np.float32([[s, 0, cx - s * cx], [0, s, cy - s * cy]])
    p.stdin.write(cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE).tobytes())
p.stdin.close(); p.wait()
