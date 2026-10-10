# Rebuild the backrest mesh window from a small clean sample so no V, seams or patches can show.
# The weave lines are near-horizontal with a slight tilt (TILT px per px). The sample (centre of the panel) is sheared
# straight, cut to a tile whose height is a whole number of weave periods and whose width repeats seamlessly, tiled over
# the hand-traced mesh WINDOW, sheared back to the original tilt and shaded like the original panel.
# Usage: python3 tools/rebuild_backrest_mesh.py in.png out.png
import os, sys, numpy as np, cv2
src, dst = sys.argv[1:3]
img = cv2.imread(src).astype(np.float32); H, W = img.shape[:2]; g = img.mean(2)
TILT = 0.0496                                            # measured: line phase drifts 1.33 rad / 50 px at period 11.73
CX, CY = 1000, 900                                       # sample centre (clean mesh)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
straight = cv2.remap(img, xx, yy + TILT * (xx - CX), cv2.INTER_LINEAR)            # lines horizontal around CX
sg = straight.mean(2)
# period: tile height = 4 periods; choose the height h (40..60) whose wrap matches best
y0 = CY - 60
h = min(range(40, 60), key=lambda h: np.abs(sg[y0:y0 + 8, CX - 120:CX + 120] - sg[y0 + h:y0 + h + 8, CX - 120:CX + 120]).mean())
x0 = CX - 150
w = min(range(150, 300), key=lambda w: np.abs(sg[y0:y0 + h, x0:x0 + 6] - sg[y0:y0 + h, x0 + w:x0 + w + 6]).mean())
tile = straight[y0:y0 + h, x0:x0 + w]
print('tile', h, w)
# tile everything (straight space), then shear back to the original tilt
reps = (H // h + 3, W // w + 3)
big = np.tile(tile, (reps[0], reps[1], 1))[:H + 2 * h, :W + 2 * w]
oy = (y0 % h); ox = (x0 % w)                          # keep the tile phase aligned with the sample position
mapx = (xx + w - ox) % (reps[1] * w); mapy = yy - TILT * (xx - CX) + 2 * h - oy
T = cv2.remap(big, mapx.astype(np.float32), mapy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
# shading: tile's own mean replaced by the original panel's smooth brightness (mesh pixels only)
mesh = (cv2.blur(g, (31, 31)) > 45).astype(np.float32)
lowT = cv2.GaussianBlur(img * mesh[..., None], (0, 0), 60) / (cv2.GaussianBlur(mesh, (0, 0), 60)[..., None] + 1e-3)
T = T / (tile.reshape(-1, 3).mean(0) + 1e-3) * lowT
# mesh opening = the original mesh areas joined across the V (closing with a large disk), so its corners and edges
# are the original's; only the V bands between them change
from scipy import ndimage
obj = ndimage.binary_fill_holes(cv2.morphologyEx((g < 200).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)))
obj = cv2.erode(obj.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0               # inside the frame's outer edge
bm = cv2.blur(g, (21, 21)); m0 = ((bm > 40) & (bm < 150) & obj).astype(np.uint8)
m0 = cv2.morphologyEx(m0, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
win = cv2.morphologyEx(m0, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (301, 301)))
from scipy import ndimage
win = ndimage.binary_fill_holes(win).astype(np.uint8)
n_, lab_, st_, _ = cv2.connectedComponentsWithStats(win); win = (lab_ == 1 + np.argmax(st_[1:, 4])).astype(np.uint8)
cnt = max(cv2.findContours(win, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
win = np.zeros((H, W), np.uint8); cv2.fillPoly(win, [cv2.convexHull(cnt)], 1)          # one clean opening, no notches
RC = 70; win = cv2.morphologyEx(win, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RC + 1, 2 * RC + 1)))
RT = int(os.environ.get('RTOP', 170))                    # top corners: larger, matching rounding (like the frame's corners)
wt = cv2.morphologyEx(win, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RT + 1, 2 * RT + 1)))
ys_ = np.where(win.any(1))[0]; ymid = (ys_.min() + ys_.max()) // 2
win[:ymid] = wt[:ymid]
win = cv2.GaussianBlur(win.astype(np.float32), (0, 0), 3); win = (win > 0.5).astype(np.uint8) * 255
win = cv2.erode(win, np.ones((3, 3), np.uint8))
a = cv2.GaussianBlur(win.astype(np.float32) / 255, (0, 0), 1.6)[..., None]
# soft shadow where the mesh tucks under the frame
dist = cv2.distanceTransform(win, cv2.DIST_L2, 5)
lip = (1 - 0.45 * np.exp(-dist / 7.0))[..., None]
T = T * lip
# old mesh left just outside the new (rounder) opening becomes frame: copy real frame from just beside it
oldmesh = cv2.dilate(((cv2.blur(g, (15, 15)) > 38)).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
ring = cv2.dilate(win, np.ones((81, 81), np.uint8)) > 0
oldm = oldmesh & ring & (cv2.dilate(win, np.ones((3, 3), np.uint8)) == 0)
frame_ok = (cv2.blur(g, (15, 15)) < 30) & ~oldmesh
base = img.copy(); cx = W // 2
ys_o, xs_o = np.where(oldm)
for dy, dxs in ((-110, 0), (-160, 0), (0, -110), (0, 110), (-70, -70), (-70, 70)):
    left = oldm[ys_o, xs_o] if dy == -110 else need
    sy = np.clip(ys_o + dy, 0, H - 1); sx = np.clip(xs_o + (dxs if xs_o.mean() < 0 else np.where(xs_o < cx, -abs(dxs), abs(dxs))), 0, W - 1)
    ok = frame_ok[sy, sx] & left
    base[ys_o[ok], xs_o[ok]] = img[sy[ok], sx[ok]]
    need = left & ~ok
fe = cv2.GaussianBlur(oldm.astype(np.float32), (0, 0), 1.5)[..., None]; base = img * (1 - fe) + base * fe
out = base * (1 - a) + T * a
cv2.imwrite(dst, np.clip(out, 0, 255).astype(np.uint8)); print('ok', dst)
