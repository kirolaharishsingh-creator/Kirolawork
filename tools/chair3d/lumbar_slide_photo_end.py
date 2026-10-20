# Lumbar USP end frame from ONE real studio photo (side 3/4 view, chair facing left): the lumbar pad + its bolt bracket
# slide DIST along the angled support arm (up-left, towards the chair's front), the arm stays bolted to the spine (its
# front end follows the bracket, like the bracket sliding in the arm's slot), the mesh backrest's lower edge rides on the
# lumbar (full shift at the contact, fading to 0 at the top of the image = its top hinge). Everything else is untouched.
#  - moving pixels: dark (chair) pixels inside hand-placed polygons (2000-px coordinates, scaled to the photo)
#  - clean plate: moving pixels removed, wall inpainted
#  - soft alpha from darkness against the clean wall, colour un-mixed, warped by a smooth displacement field
# Usage: python lumbar_slide_photo_end.py photo.jpg outdir   (env: DIST 40 px @2000 wide, OUT_W 1920,3840)
import cv2, numpy as np, sys, os
src, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
I = cv2.imread(src).astype(np.float32); H, W = I.shape[:2]; k = W / 2000
P = lambda pts: (np.array(pts, np.float32) * k).astype(np.int32)
DIST = float(os.environ.get('DIST', 40)) * k
ARM_A, ARM_B = np.array([1500, 318.]) * k, np.array([1748, 465.]) * k    # bracket bolt -> spine bolt
ARM_J = np.array([1650, 400.]) * k                                       # arm crosses the rear frame bar: static from here
u = (ARM_B - ARM_A) / np.linalg.norm(ARM_B - ARM_A); S = -u * DIST       # slide: along the arm, away from the spine
# regions (2000-px coords): lumbar + bracket + backrest bottom; the arm separately
MOVE = P([(1240, 0), (1612, 0), (1590, 200), (1572, 262), (1566, 365), (1490, 375), (1540, 470), (1600, 540), (1660, 610),
          (1690, 672), (1694, 694), (1705, 704), (1708, 714), (1704, 724), (1695, 728), (1650, 735),   # tip outline (inset)
          (1600, 718), (1550, 700), (1500, 670), (1450, 636), (1400, 600), (1356, 570), (1240, 540)])          # lower edge: mid-gap lumbar|seat (measured)
ARM = P([(1530, 300), (1600, 300), (1700, 400), (1765, 450), (1760, 505), (1700, 505), (1570, 410), (1530, 360)])
STATIC = P([(1628, 0), (2000, 0), (2000, 800), (1740, 800), (1713, 700), (1703, 693), (1722, 650), (1735, 600), (1700, 520), (1688, 420),
            (1680, 370), (1660, 280), (1640, 150)])          # rear frame bar + spine: never moved, never filled
x0, y0, x1, y1 = int(1180 * k), 0, int(1830 * k), int(800 * k)              # work crop
C = I[y0:y1, x0:x1].copy(); h, w = C.shape[:2]
def poly(p): m = np.zeros((h, w), np.uint8); cv2.fillPoly(m, [p - [x0, y0]], 1); return m
L = cv2.cvtColor(C.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
dark = (L < 140).astype(np.uint8)
mv, ar = poly(MOVE) & dark, poly(ARM) & dark & (1 - poly(MOVE))
stat = poly(STATIC)
mv, ar = mv & (1 - stat), ar & (1 - stat)
movers = mv | ar
grow = cv2.dilate(movers, np.ones((int(9 * k) | 1,) * 2, np.uint8)) & (1 - stat)
# clean plate: everything uncovered is studio wall - fill it from wall pixels only (normalized convolution), so no
# dark spine / seat pixels bleed in; light grain added back
wallm = ((L > 150) & (grow == 0)).astype(np.float32); sg = 25 * k
num = cv2.GaussianBlur(C * wallm[..., None], (0, 0), sg); den = cv2.GaussianBlur(wallm, (0, 0), sg)[..., None]
wall = num / np.maximum(den, 1e-4) + np.random.default_rng(2).normal(0, 1.5, C.shape)
clean = np.where(grow[..., None] > 0, wall, C)
# except under the lumbar tip: it sat in front of the spine's lower piece, whose upper edge runs (1666,733)->(1700,697)
# (2000-px); the uncovered part on the spine side of that line is spine black, not wall
yy0, xx0 = np.mgrid[0:h, 0:w].astype(np.float32); Xg, Yg = (xx0 + x0) / k, (yy0 + y0) / k
sp = ((Xg - 1666) * 36 + (Yg - 733) * 34 > 0) & (Xg > 1630) & (Xg < 1730) & (Yg > 680) & (Yg < 770)
# fill it with real spine texture cloned from further along that same edge (down-left, where the piece is visible)
v = np.array([-37.0, 40.0]) * k
srcx = np.clip(xx0 + v[0], 0, w - 1).astype(np.float32); srcy = np.clip(yy0 + v[1], 0, h - 1).astype(np.float32)
clone = cv2.remap(C.astype(np.float32), srcx, srcy, cv2.INTER_LINEAR)
hole_sp = ((grow > 0) & sp)
dk = np.median(C[(stat > 0) & (L < 60) & (Xg > 1705) & (Xg < 1740) & (Yg > 700) & (Yg < 760)], axis=0)
clone = np.where((cv2.cvtColor(np.clip(clone, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) > 90)[..., None], dk, clone)
clean = np.where(hole_sp[..., None], clone, clean)          # hard edge: no wall-colour rim against the spine
# soft alpha + un-mixed colour
bgL = cv2.cvtColor(clean.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
fgL = np.percentile(L[movers > 0], 20)
a = np.clip((bgL - L) / np.maximum(bgL - fgL, 1), 0, 1) * grow
# only the traced outlines (plus a hair for the anti-aliased edge against the wall) - never neighbouring spine pixels
a = a * (cv2.dilate(poly(MOVE) | poly(ARM), np.ones((int(3 * k) | 1,) * 2, np.uint8)) & (1 - stat))
a = np.where(cv2.erode(movers, np.ones((3, 3), np.uint8)) > 0, 1.0, a)
# lumbar tip: it sat on the black spine, so its edge pixels carry spine black - use a soft edge from the inset outline
tipb = ((Xg > 1655) & (Xg < 1722) & (Yg > 680) & (Yg < 745)).astype(np.float32)
soft = cv2.GaussianBlur(cv2.erode(poly(MOVE), np.ones((int(2 * k) | 1,) * 2, np.uint8)).astype(np.float32), (0, 0), 1.5 * k)
a = a * (1 - tipb) + np.minimum(a, soft) * tipb
F = (C - (1 - a[..., None]) * clean) / np.maximum(a[..., None], 0.05)
# displacement weight at OUTPUT positions: 1 on the lumbar, fades to 0 at the top (backrest hinge), linear along the arm
yy, xx = np.mgrid[0:h, 0:w].astype(np.float32); X, Y = xx + x0, yy + y0
wgt = np.clip(Y / (150 * k), 0, 1); wgt = wgt * wgt * (3 - 2 * wgt)
wgt = wgt * (cv2.dilate(poly(MOVE), np.ones((int(25 * k) | 1,) * 2, np.uint8)) > 0)    # nothing else moves
armz = cv2.dilate(poly(ARM) & (1 - poly(MOVE)), np.ones((int(15 * k) | 1,) * 2, np.uint8)) > 0
t = np.clip(((X - ARM_J[0]) * -u[0] + (Y - ARM_J[1]) * -u[1]) / np.linalg.norm(ARM_J - ARM_A), 0, 1)
wgt = np.where(armz, t, wgt)
wgt = cv2.GaussianBlur(wgt.astype(np.float32), (0, 0), 2 * k)
mx, my = (xx - S[0] * wgt).astype(np.float32), (yy - S[1] * wgt).astype(np.float32)
a2 = cv2.remap(a.astype(np.float32), mx, my, cv2.INTER_CUBIC, borderValue=0)
F2 = cv2.remap(F.astype(np.float32), mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
a2 = np.clip(a2, 0, 1)[..., None]
E = I.copy(); E[y0:y1, x0:x1] = F2 * a2 + clean * (1 - a2)
E = np.clip(E, 0, 255).astype(np.uint8); Su = np.clip(I, 0, 255).astype(np.uint8)
print('photo', W, H, 'shift px', S.round(1), 'moving px', int(movers.sum()))
for ow in [int(v) for v in os.environ.get('OUT_W', '1920,3840').split(',')]:
    oh = ow * 9 // 16
    for nm, im in (('start', Su), ('end', E)):
        cv2.imwrite('%s/lumbar_photo_%s_%d.png' % (out, nm, ow), cv2.resize(im, (ow, oh), interpolation=cv2.INTER_AREA))
cv2.imwrite(out + '/lumbar_photo_end_full.jpg', E, [cv2.IMWRITE_JPEG_QUALITY, 95])
