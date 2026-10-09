# Lumbar USP loop from one photo-real still (the cleaned Nano Banana image, spine left / seat right): the lumbar pad is
# cut out and rocked about its pivot bolt -- lower edge swinging forward toward the seat, a short hold, back to rest --
# as in the reference video (about 5 degrees). What the pad uncovers is inpainted from its surroundings; the backrest
# piece above the pad stays in front. Frame N equals frame 0, so it loops.
# Usage: python3 lumbar_swing_photo.py still.jpg pad_mask.npy outdir   (env: ANGLE deg (5), FPS (24), SECONDS (2.5))
# Known issues (first test): a light smudge left where the pad tip was by the spine bracket, and a small flat cut in
# the seat top edge where the swinging pad crosses it.
import os, sys, numpy as np, cv2
src, mpath, out = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(out, exist_ok=True)
ANG = float(os.environ.get('ANGLE', 5)); FPS = int(os.environ.get('FPS', 24)); SEC = float(os.environ.get('SECONDS', 2.5))
PIV = (563.0, 560.0)                                                   # pivot bolt (2000 x 1125 image)
im = cv2.imread(src).astype(np.float32); H, W = im.shape[:2]
pad = np.load(mpath).astype(np.uint8)
pad = cv2.morphologyEx(pad, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
cnt, _ = cv2.findContours(pad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); pad = np.zeros_like(pad); cv2.drawContours(pad, cnt, -1, 1, -1)
a = cv2.GaussianBlur(pad.astype(np.float32), (0, 0), 0.8)
# plate without the pad: inpaint from the surroundings (backdrop, seat edge, spine)
plate = cv2.inpaint(im.astype(np.uint8), cv2.dilate(pad, np.ones((5, 5), np.uint8)) * 255, 7, cv2.INPAINT_TELEA).astype(np.float32)
# the backrest above the pad top stays in front of it
front = np.zeros((H, W), np.uint8)
cv2.fillPoly(front, [np.array([(470, 250), (700, 250), (700, 372), (640, 362), (498, 366), (470, 380)], np.int32)], 1)
front = cv2.GaussianBlur(((front > 0) & (im.mean(2) < 120) & (pad == 0)).astype(np.float32), (0, 0), 0.8)[..., None]
N = int(round(SEC * FPS))
def ease(u): return u * u * (3 - 2 * u)
def angle(i):                                                           # rest -> forward -> hold -> back -> rest
    u = i / N; t1, t2, t3 = 0.40, 0.52, 0.92
    if u < t1: return ANG * ease(u / t1)
    if u < t2: return ANG
    if u < t3: return ANG * (1 - ease((u - t2) / (t3 - t2)))
    return 0.0
for i in range(N):
    M = cv2.getRotationMatrix2D(PIV, float(os.environ.get('SIGN', 1)) * angle(i), 1.0)
    L = cv2.warpAffine(im, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    ra = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR)[..., None]
    f = L * ra + plate * (1 - ra)
    f = im * front + f * (1 - front)
    cv2.imwrite(f'{out}/{i:04d}.png', np.clip(f + 0.5, 0, 255).astype(np.uint8))
print('ok', N, 'frames')
