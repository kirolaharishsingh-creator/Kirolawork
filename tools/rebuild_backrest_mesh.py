# Rebuild the backrest mesh window from a small clean sample so no V, seams or patches can show.
# The weave lines are near-horizontal with a slight tilt (TILT px per px). The sample (centre of the panel) is sheared
# straight, cut to a tile whose height is a whole number of weave periods and whose width repeats seamlessly, tiled over
# the hand-traced mesh WINDOW, sheared back to the original tilt and shaded like the original panel.
# Usage: python3 tools/rebuild_backrest_mesh.py in.png out.png
import sys, numpy as np, cv2
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
WINDOW = [(395, 282), (1565, 292), (1697, 402), (1702, 1000), (1748, 1600), (1705, 1700), (1300, 1712), (840, 1800),
          (480, 1800), (410, 950), (330, 560)]
win = np.zeros((H, W), np.uint8); cv2.fillPoly(win, [np.int32(WINDOW)], 255); win = cv2.erode(win, np.ones((5, 5), np.uint8))
a = cv2.GaussianBlur(win.astype(np.float32) / 255, (0, 0), 2.0)[..., None]
out = img * (1 - a) + T * a
cv2.imwrite(dst, np.clip(out, 0, 255).astype(np.uint8)); print('ok', dst)
