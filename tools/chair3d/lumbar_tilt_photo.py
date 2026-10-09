# End frame for the lumbar USP from the real side photo (photo 04 on the studio backdrop): the C-shaped lumbar pad
# (with its bracket) is cut out, rotated about its pivot bolt so its lower limb swings forward (toward the seat front,
# as in the reference video), and pasted back. What the pad uncovers is studio backdrop (the
# seat and spine lie outside its outline); the backrest above, the pivot arm and the spine are
# drawn back on top, since they sit in front of or beside the pad.
# Usage (python with cv2): python3 lumbar_tilt_photo.py start.jpg out.jpg   (env: ANGLE clockwise deg, 8)
import os, sys, numpy as np, cv2
from PIL import Image, ImageDraw

src, dst = sys.argv[1], sys.argv[2]
ANG = float(os.environ.get('ANGLE', 8)); PIV = (2205, 455)          # pivot bolt, pixels at 2880 x 1620
f = np.asarray(Image.open(src).convert('RGB'), np.float32) / 255; H, W = f.shape[:2]; assert (W, H) == (2880, 1620)
lum = f.mean(-1) * 255
def poly(pts):
    m = Image.new('L', (W, H), 0); ImageDraw.Draw(m).polygon(pts, 255); return np.asarray(m) > 0
k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
# the pad: outer C edge, its top under the backrest, the bracket around the bolt, the lower limb down to its tail
PAD = [(1790, 430), (1840, 300), (1930, 222), (2030, 200), (2150, 228), (2262, 262), (2282, 300), (2282, 395),
       (2250, 420), (2160, 540), (2290, 700), (2440, 830), (2462, 1000), (2462, 1040), (2440, 1052), (2300, 985),
       (2150, 955), (1960, 880), (1850, 770), (1795, 600)]
yy, xx = np.mgrid[0:H, 0:W]; BRK = np.hypot(xx - PIV[0], yy - PIV[1]) < 54          # the round bracket on the bolt
pad = (poly(PAD) | BRK) & (lum < 120)
pad = cv2.morphologyEx(pad.astype(np.uint8), cv2.MORPH_OPEN, k(2))
n, lab, st, _ = cv2.connectedComponentsWithStats(pad); pad = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
pad = cv2.morphologyEx(pad, cv2.MORPH_CLOSE, k(4))
# fill the pad's own dark mesh lines / shadows inside its outline (holes smaller than the C's opening)
cnts, hier = cv2.findContours(pad, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
for i, c in enumerate(cnts):
    if hier[0][i][3] >= 0 and cv2.contourArea(c) < 4000: cv2.drawContours(pad, [c], -1, 1, -1)
a = cv2.GaussianBlur(pad.astype(np.float32), (0, 0), 0.9)
# base: the pad removed; behind it there is only backdrop (the seat and spine are outside the outline)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comp import studio_bg
# (the bracket turns in place and the pad's top-right corner sits against the backrest: keep the photo under both)
keep = np.hypot(xx - PIV[0], yy - PIV[1]) < 58
hole = cv2.GaussianBlur((cv2.dilate(pad, k(3)) & ~keep).astype(np.float32), (0, 0), 1.2)[..., None]
base = f * (1 - hole) + studio_bg(W, H).astype(np.float32) * hole
# behind the pad's top-right corner is the underside of the backrest: fill that corner with the backrest's colour
CORNER = poly([(2200, 240), (2300, 262), (2300, 340), (2200, 300)]) & (cv2.dilate(pad, k(3)) > 0)
cb = cv2.GaussianBlur(CORNER.astype(np.float32), (0, 0), 1.5)[..., None]
base = base * (1 - cb) + np.median(f[215:240, 2180:2240].reshape(-1, 3), 0) * cb
# by the spine the tail uncovers the spine's own left edge (hidden behind the tail in the photo): continue that edge
# as a smooth curve and fill the uncovered part right of it with the spine's colour, backdrop left of it
SPINE_EDGE = [(2505, 820), (2495, 950), (2468, 1030), (2425, 1100), (2400, 1140)]
spine_fill = poly(SPINE_EDGE + [(2880, 1140), (2880, 820)])
sc = np.median(f[940:1000, 2505:2530].reshape(-1, 3), 0)
sh = (cv2.dilate(pad, k(3)) > 0) & spine_fill & ~keep
sb = cv2.GaussianBlur(sh.astype(np.float32), (0, 0), 1.2)[..., None]
base = base * (1 - sb) + (f * 0 + sc) * sb
# rotate the pad layer about the bolt (cv2 positive angle = counter-clockwise on screen)
M = cv2.getRotationMatrix2D(PIV, -ANG, 1.0)
rimg = cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
ra = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR)[..., None]
out = rimg * ra + base * (1 - ra)
# in front of the pad: the backrest's lower edge above the pad top, the pivot arm beyond the bracket, the spine
FRONT = poly([(1940, 60), (2330, 60), (2330, 300), (2262, 290), (2252, 268), (2150, 226), (2030, 204), (1958, 214)]) \
      | (poly([(2262, 400), (2560, 560), (2560, 720), (2230, 530)]) & (lum < 40) & ~BRK) \
      | ((poly([(2462, 0), (2880, 0), (2880, 820), (2462, 820)]) | spine_fill | poly([(2300, 1140), (2880, 1140), (2880, 1620), (2300, 1620)])) & (lum < 120))
fr = cv2.GaussianBlur((FRONT & (lum < 120) & ~((pad > 0) & spine_fill)).astype(np.float32), (0, 0), 0.8)[..., None]
out = f * fr + out * (1 - fr)
Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)).save(dst, quality=95)
Image.fromarray(pad * 255).save(dst.rsplit('.', 1)[0] + '_mask.png')
print('ok', dst, 'angle', ANG)
