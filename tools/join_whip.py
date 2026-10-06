# Join clips back to back; where marked, a whip pan (fast slide + horizontal motion blur) spans the cut.
# Usage: python3 tools/join_whip.py out.mp4 clip1.mp4 [+]clip2.mp4 [+]clip3.mp4 ...   ('+' = whip into this clip, else hard/seamless cut)
import sys, subprocess, cv2, numpy as np
W, H, HALF = 1920, 1080, 4
MAX_SHIFT, MAX_BLUR = 0.45 * W, 220
def whip(f, k, sign):
    e = k ** 2
    f = cv2.warpAffine(f, np.float32([[1, 0, sign * e * MAX_SHIFT], [0, 1, 0]]), (W, H), borderMode=cv2.BORDER_REPLICATE)
    b = int(e * MAX_BLUR)
    return cv2.blur(f, (b, 1)) if b > 2 else f
clips, whips = [], []
for arg in sys.argv[2:]:
    whips.append(arg.startswith('+')); cap = cv2.VideoCapture(arg.lstrip('+')); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    clips.append(fr)
p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', '24', '-i', '-',
                      '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', sys.argv[1]], stdin=subprocess.PIPE)
sign = -1
for c, fr in enumerate(clips):
    for i, f in enumerate(fr):
        if c + 1 < len(clips) and whips[c + 1] and i >= len(fr) - HALF:      # leaving into a whip
            f = whip(f, (i - (len(fr) - HALF) + 1) / HALF, sign)
        if whips[c] and i < HALF:                                            # arriving from a whip
            f = whip(f, (HALF - i) / HALF, -sign)
        p.stdin.write(f.tobytes())
    if c + 1 < len(clips) and whips[c + 1]: sign = -sign
p.stdin.close(); p.wait()
print(sum(len(f) for f in clips), 'frames')
