# Anatomy edit in the reference style: shots joined by whip pans (fast sideways slide + horizontal motion blur).
# Same shots, trims, colour match and shot 3 line-cooling as assemble_anatomy.py.
# Usage: python3 tools/assemble_whip.py out.mp4
import cv2, numpy as np, subprocess, sys
D = 'anatomy_edit/'
SHOTS = [  # file, frames to keep, take from end?
    ('shot1_swivel_cut', 30, False), ('shot2_headrest_cut', 29, False), ('shot3_mesh_cut_v3', 30, False),
    ('shot4_lumbar_cut', 30, False), ('shot5_side_cut', 30, False), ('shot6_armrest_cut', 30, False),
    ('shot7_mechanism_cut_v3', 30, False), ('shot8_hero_rear_cut', 31, True)]
COOL = {'shot3_mesh_cut_v3'}
W, H, HALF = 1920, 1080, 4            # whip spans 4 frames out of one shot + 4 frames into the next
MAX_SHIFT, MAX_BLUR = 0.45 * W, 220   # at the cut: slide 45% of the width, 220 px streak

def frames(name):
    cap = cv2.VideoCapture(D + name + '.mp4'); out = []
    while True:
        ok, f = cap.read()
        if not ok: return out
        out.append(f)

def cool(f):
    lum = f.mean(2); onchair = cv2.blur((lum < 60).astype(np.float32), (31, 31))
    w = (np.clip((lum - 120) / 60, 0, 1) * np.clip((onchair - 0.6) / 0.15, 0, 1))[..., None]
    return f * (1 + w * np.array([0.12, 0.03, -0.08]))

def whip(f, k, sign):
    # k: 0..1 strength (1 at the cut). Ease-in so it snaps fast near the cut.
    e = k ** 2
    shift, blur = sign * e * MAX_SHIFT, int(e * MAX_BLUR)
    M = np.float32([[1, 0, shift], [0, 1, 0]])
    f = cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    if blur > 2: f = cv2.blur(f, (blur, 1))
    return f

clips, bgs = [], []
for name, n, from_end in SHOTS:
    fr = frames(name); fr = fr[-n:] if from_end else fr[:n]
    f0 = fr[0].astype(np.float32); c = np.concatenate([f0[:150, :250].reshape(-1, 3), f0[:150, -250:].reshape(-1, 3)])
    bgs.append(c[c.mean(1) > np.percentile(c.mean(1), 80)].mean(0)); clips.append(fr)
target = np.median(bgs, 0)
gains = [np.clip(target / b, 0.92, 1.08) for b in bgs]
seq = [(f, g, name in COOL) for c, g, (name, _, _) in zip(clips, gains, SHOTS) for f in c]
cuts = list(np.cumsum([len(c) for c in clips])[:-1])

p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', '24', '-i', '-',
                      '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', sys.argv[1]], stdin=subprocess.PIPE)
for i, (f, g, c) in enumerate(seq):
    f = f.astype(np.float32) * g
    if c: f = cool(f)
    for k, cut in enumerate(cuts):
        sign = -1 if k % 2 == 0 else 1          # alternate whip direction
        j = i - cut                              # -HALF..-1 = leaving shot, 0..HALF-1 = arriving shot
        if -HALF <= j < 0: f = whip(f, (j + HALF + 1) / HALF, sign)            # outgoing: slide away
        elif 0 <= j < HALF: f = whip(f, (HALF - j) / HALF, -sign)              # incoming: enters from the other side, same pan direction
    p.stdin.write(np.clip(f, 0, 255).astype(np.uint8).tobytes())
p.stdin.close(); p.wait()
print(len(seq), 'frames', len(seq) / 24, 's')
