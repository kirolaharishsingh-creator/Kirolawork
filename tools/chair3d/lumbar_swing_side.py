# Lumbar USP loop from the real side photo 04 on the studio backdrop (seat left, spine right): the lumbar pad's lower
# part swings forward (toward the seat, i.e. left) on its pivot bolt, holds briefly, and swings back, as in the
# reference video (about 6 degrees at the tip, lower part forward and slightly down). The mount around the bolt stays
# put and the turn grows with distance from it, so the top barely moves. Frame N equals frame 0, so it loops.
# Behind the pad the plate is rebuilt: the studio backdrop, and at the tip the spine bracket's dark slot inside its two
# upper-left edge. The seat and the backrest above the pad stay in front.
# Usage: python3 lumbar_swing_side.py still.png outdir   (still: start_frame_lumbar_studio.jpg at 2000 x 1125)
#        env: ANGLE deg (6), FPS (24), SECONDS (2.5)
import os, sys, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comp import studio_bg
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
ANG = float(os.environ.get('ANGLE', 6)); FPS = int(os.environ.get('FPS', 24)); SEC = float(os.environ.get('SECONDS', 2.5))
PIV = (1529.0, 316.0)                                                   # pivot bolt
im = cv2.imread(src).astype(np.float32); H, W = im.shape[:2]; lum = im.mean(2); assert (W, H) == (2000, 1125)
Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
poly = lambda pts: cv2.fillPoly(np.zeros((H, W), np.uint8), [np.array(pts, np.int32)], 1) > 0
k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
bg = (studio_bg(W, H)[..., ::-1] * 255).astype(np.float32)

# the pad: outer C edge, inner edge, and along the thin backdrop gap above the seat; everything darker than the backdrop
PAD = [(1258, 330), (1281, 212), (1330, 156), (1390, 134), (1440, 144), (1485, 176), (1505, 215), (1470, 240), (1425, 258),
       (1420, 330), (1424, 420), (1446, 490), (1518, 555), (1598, 605), (1680, 646), (1712, 656), (1704, 700), (1700, 734),
       (1650, 738), (1600, 725), (1550, 700), (1500, 670), (1450, 636), (1400, 600), (1350, 560), (1300, 505), (1262, 420)]
pad = (poly(PAD) & (lum < 185) & ~((X > 1640) & (Y > 640) & (lum < 20))).astype(np.uint8)   # not the slot
pad = cv2.morphologyEx(pad, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
# at the tip, the slot's anti-aliased upper-left edge above the tip and the rib frame's rim below it are not pad
pad[(X > 1696 - 0.6 * (Y - 690) - 2) & (Y < 698) & (X > 1640)] = 0
pad[(Y > 734) & (X > 1650)] = 0; pad[(Y > 737) & (X > 1620)] = 0
# the tip's rounded end itself (it overlaps the slot, so it is outlined rather than thresholded)
pad[poly([(1640, 660), (1665, 667), (1682, 677), (1694, 689), (1703, 701), (1707, 713), (1704, 722), (1696, 729),
          (1680, 731), (1640, 734)])] = 1
n, lab, st, _ = cv2.connectedComponentsWithStats(pad); pad = (lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8)
a = cv2.GaussianBlur(pad.astype(np.float32), (0, 0), 0.7)

# plate behind the pad: backdrop; at the tip, the bracket's dark slot right of its upper-left edge; around the bolt (thin slivers only) inpainted from the mount and arm
xin = lambda y: 1702 - 0.6 * (y - 690)                                 # slot's upper-left edge (1736,632)..(1655,757)
tipzone = (X > 1630) & (Y > 620) & (Y < 800)
dark = np.clip(xin(Y) - X + 0.5, 0, 1)[..., None] * 0 + np.clip(X - xin(Y) + 0.5, 0, 1)[..., None]
syn = np.where(tipzone[..., None], np.array([7.0, 7.0, 8.0], np.float32) * dark + bg * (1 - dark), bg)
inp = cv2.inpaint(np.clip(im, 0, 255).astype(np.uint8), cv2.dilate(pad, k(2)) * 255, 5, cv2.INPAINT_TELEA).astype(np.float32)
rr = np.hypot(X - PIV[0], Y - PIV[1])
syn = np.where((rr < 175)[..., None], inp, syn)
R = cv2.GaussianBlur(cv2.dilate(pad, k(3)).astype(np.float32), (0, 0), 1.0)[..., None]
plate = im * (1 - R) + syn * R
# in front of the pad: the seat (black fabric below the gap) and the backrest piece above the pad's top end
seat = poly([(1100, 520), (1350, 565), (1400, 605), (1450, 641), (1500, 676), (1550, 706), (1600, 731), (1640, 760),
             (1640, 1125), (1100, 1125)]) & (lum < 30)
front = poly([(1380, 60), (1600, 60), (1600, 200), (1485, 176), (1440, 144), (1390, 134), (1380, 120)]) & (lum < 120) & (pad == 0)
F = cv2.GaussianBlur((seat | front).astype(np.float32), (0, 0), 0.7)[..., None]

dX, dY = X - PIV[0], Y - PIV[1]
R0, R1 = 110.0, 300.0
wr = np.clip((rr - R0) / (R1 - R0), 0, 1); wr = wr * wr * (3 - 2 * wr)
N = int(round(SEC * FPS)); ease = lambda u: u * u * (3 - 2 * u)
def angle(i):                                                           # rest -> forward -> hold -> back -> rest
    u = i / N; t1, t2, t3 = 0.40, 0.52, 0.92
    if u < t1: return ANG * ease(u / t1)
    if u < t2: return ANG
    if u < t3: return ANG * (1 - ease((u - t2) / (t3 - t2)))
    return 0.0
for i in range(N):
    th = -np.radians(angle(i)) * wr                                    # clockwise on screen: lower part moves left
    c_, s_ = np.cos(th), np.sin(th)
    mx = (PIV[0] + c_ * dX - s_ * dY).astype(np.float32); my = (PIV[1] + s_ * dX + c_ * dY).astype(np.float32)
    Lr = cv2.remap(im, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    ra = cv2.remap(a, mx, my, cv2.INTER_LINEAR)[..., None]
    f = Lr * ra + plate * (1 - ra)
    f = im * F + f * (1 - F)
    if i == 0: f = im
    cv2.imwrite(f'{out}/{i:04d}.png', np.clip(f + 0.5, 0, 255).astype(np.uint8))
cv2.imwrite(f'{out}/plate.png', np.clip(plate + 0.5, 0, 255).astype(np.uint8))
print('ok', N, 'frames')
