# Smooth push-in (Ken Burns) on a 4K still, rendered to 1080p. Starts on the full frame so it can
# continue straight from a Kling clip that ends on the same keyframe.
# Usage: python3 tools/push_in.py still.jpg out.mp4 cx cy crop_h seconds [start_scale]
# start_scale > 1 starts already zoomed in on the frame centre (to match a clip that ended zoomed).
import sys, subprocess, cv2, numpy as np
src = cv2.imread(sys.argv[1]); H0, W0 = src.shape[:2]
cx, cy, ch, secs = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
W, H, FPS = 1920, 1080, 24
n = int(round(secs * FPS))
s0 = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
h_start = H0 / s0
p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                      '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', sys.argv[2]], stdin=subprocess.PIPE)
for i in range(n):
    t = i / (n - 1)
    e = t * t * (3 - 2 * t)                                # ease in and out
    h = h_start + (ch - h_start) * e; w = h * W / H
    x = (W0 / 2) + (cx - W0 / 2) * e - w / 2; y = (H0 / 2) + (cy - H0 / 2) * e - h / 2
    M = np.float32([[W / w, 0, -x * W / w], [0, H / h, -y * H / h]])
    p.stdin.write(cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_AREA if w > W else cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE).tobytes())
p.stdin.close(); p.wait()
