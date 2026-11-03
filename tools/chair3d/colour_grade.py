# Colour grade only (no compositing, no retiming): match a clip to the premium light look of premium_look.py.
#  - one fixed grade per clip (no flicker): the wall colour is measured once (median of the light pixels over frames
#    sampled across the clip) and per-channel LUTs map it to the cool clean white of the premium backdrop
#  - tone: black point + gentle S-curve (deeper blacks, slightly more contrast), highlights kept
# Usage: python colour_grade.py in.mp4 out.mp4   (env: WALL r,g,b target 236,238,242  BP 0.02  GAMMA 1.10  SAT 0.92)
import os, sys, subprocess, numpy as np, cv2
src, out = sys.argv[1:3]
T = np.array([float(v) for v in os.environ.get('WALL', '236,238,242').split(',')])[::-1]       # BGR
BP, GAM, SAT = float(os.environ.get('BP', 0.02)), float(os.environ.get('GAMMA', 1.10)), float(os.environ.get('SAT', 0.92))
cap = cv2.VideoCapture(src); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
samp = []
for i in np.linspace(0, n - 1, 12).astype(int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = cap.read()
    if ok:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); samp.append(f[g > np.percentile(g, 70)])
wall = np.median(np.concatenate(samp), axis=0)                                                 # BGR
x = np.arange(256) / 255.0
def tone(v):
    v = np.clip((v - BP) / (1 - BP), 0, 1)
    s = v * v * (3 - 2 * v); v = 0.75 * v + 0.25 * s                                          # gentle S
    return v ** GAM
luts = []
for c in range(3):
    tw = tone(wall[c] / 255.0); gain = (T[c] / 255.0) / max(tw, 1e-3)                          # wall lands on target
    y = tone(x) * gain; y = np.where(y > 0.9, 0.9 + (1 - 0.9) * np.tanh((y - 0.9) / 0.1), y)   # soft highlight roll-off
    luts.append(np.clip(y * 255, 0, 255).astype(np.uint8))
lut = np.stack(luts, 1)[:, None, :]                                                            # 256x1x3
print('wall measured BGR', wall.round(1), '-> target', T)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
p = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', f'{fps}', '-i', '-',
                      '-i', src, '-map', '0:v', '-map', '1:a?', '-c:a', 'copy', '-shortest',
                      '-c:v', 'libx264', '-crf', '14', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out],
                     stdin=subprocess.PIPE)
k = 0
while True:
    ok, f = cap.read()
    if not ok: break
    g = cv2.LUT(f, lut)
    if SAT != 1:                                                                               # calmer, neutral colour
        hsv = cv2.cvtColor(g, cv2.COLOR_BGR2HSV).astype(np.float32); hsv[..., 1] *= SAT
        g = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    p.stdin.write(g.tobytes()); k += 1
p.stdin.close(); p.wait(); print('ok', out, k, 'frames @', fps)
