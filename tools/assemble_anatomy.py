# Assemble the 10 s anatomy edit: trim to 240 frames, match background colour, light sweep across each cut.
import cv2, numpy as np, subprocess, sys
D = 'anatomy_edit/'
SHOTS = [  # file, frames to keep, take from end?
    ('shot1_swivel_cut', 30, False), ('shot2_headrest_cut', 29, False), ('shot3_mesh_cut_v3', 30, False),
    ('shot4_lumbar_cut', 30, False), ('shot5_side_cut', 30, False), ('shot6_armrest_cut', 30, False),
    ('shot7_mechanism_cut_v3', 30, False), ('shot8_hero_rear_cut', 31, True)]
W, H, SWEEP = 1920, 1080, 8                      # sweep spans 8 frames centred on each cut

def frames(name):
    cap = cv2.VideoCapture(D + name + '.mp4'); out = []
    while True:
        ok, f = cap.read()
        if not ok: return out
        out.append(f)

clips, bgs = [], []
for name, n, from_end in SHOTS:
    fr = frames(name); fr = fr[-n:] if from_end else fr[:n]
    # background sample: brightest 20% of the top-left and top-right corners of the first frame
    f0 = fr[0].astype(np.float32); c = np.concatenate([f0[:150, :250].reshape(-1, 3), f0[:150, -250:].reshape(-1, 3)])
    bgs.append(c[c.mean(1) > np.percentile(c.mean(1), 80)].mean(0)); clips.append(fr)
target = np.median(bgs, 0)                       # match every shot to the shared studio grey
gains = [np.clip(target / b, 0.92, 1.08) for b in bgs]
for (name, _, _), b, g in zip(SHOTS, bgs, gains): print(name, np.round(b), 'gain', np.round(g, 3))

seq = [(f, g) for c, g in zip(clips, gains) for f in c]
cuts = list(np.cumsum([len(c) for c in clips])[:-1])
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
def sweep(i):
    for k, cut in enumerate(cuts):
        j = i - cut
        if -SWEEP // 2 <= j < SWEEP // 2:
            sign = 1 if k % 2 == 0 else -1          # alternate sweep direction with the camera moves
            t = (j + SWEEP / 2 + 0.5) / SWEEP       # 0..1 across the sweep
            pos = (t if sign > 0 else 1 - t) * (W + 600) - 300
            band = np.exp(-(((xx + 0.35 * yy) - pos - 0.17 * H) / 40.0) ** 2)   # thin diagonal light streak
            return band * 0.6 * np.sin(np.pi * t)
    return None

p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', '24', '-i', '-',
                      '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', sys.argv[1]], stdin=subprocess.PIPE)
for i, (f, g) in enumerate(seq):
    f = f.astype(np.float32) * g
    b = sweep(i)
    if b is not None: f = f + (255 - f) * b[..., None]
    p.stdin.write(np.clip(f, 0, 255).astype(np.uint8).tobytes())
p.stdin.close(); p.wait()
print(len(seq), 'frames', len(seq) / 24, 's')
