# Remove the dark V (the Y spine seen through the mesh) from a backrest image, so the panel reads as plain mesh.
# WINDOW is the hand-traced outer edge of the mesh panel (Nano Banana backrest image, 2048 px): everything inside it that
# is not mesh gets mesh. The weave runs in near-horizontal lines, so each dark run in a row is filled with the mesh from
# the same rows beside it (left and/or right, whichever really is mesh), shifted vertically per 40-row slab so the weave
# lines stay continuous at that seam, and cross-faded across the run.
# Usage: python3 tools/remove_backrest_v.py in.png out.png [debug.jpg]
import sys, numpy as np, cv2

src, dst = sys.argv[1:3]; dbg = sys.argv[3] if len(sys.argv) > 3 else None
img = cv2.imread(src).astype(np.float32); g = img.mean(2); H, W = g.shape
mesh = cv2.blur(g, (31, 31)) > 45
mesh = cv2.morphologyEx(mesh.astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)) > 0
WINDOW = [(395, 282), (1565, 292), (1697, 402), (1702, 1000), (1748, 1600), (1705, 1700), (1300, 1712), (840, 1800),
          (480, 1800), (410, 950), (330, 560)]
win = np.zeros((H, W), np.uint8); cv2.fillPoly(win, [np.int32(WINDOW)], 1)
win = cv2.erode(win, np.ones((7, 7), np.uint8)) > 0
fill = win & ~cv2.dilate(mesh.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
n, lab, st, _ = cv2.connectedComponentsWithStats(fill.astype(np.uint8))
fill = np.isin(lab, [i for i in range(1, n) if st[i, 4] > 3000])
fill = cv2.dilate(fill.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & win
out = img.copy(); SEAM, HALF = 16, 20

def runs(row):
    xs = np.where(row)[0]
    if not len(xs): return []
    cut = np.where(np.diff(xs) > 1)[0]
    return [(xs[s], xs[e]) for s, e in zip(np.r_[0, cut + 1], np.r_[cut, len(xs) - 1])]

def shift(y, xt, xs_):
    # sub-pixel vertical shift d so img(y'+d, xs_) matches img(y', xt) over rows y-HALF..y+HALF
    ys = np.arange(max(y - HALF, 0), min(y + HALF, H - 7)); errs = []
    for dy in range(-6, 7):
        yy = np.clip(ys + dy, 0, H - 1); errs.append(np.abs(g[np.ix_(ys, xt)] - g[np.ix_(yy, xs_)]).mean())
    errs = np.array(errs); k = int(errs.argmin())
    if 0 < k < 12:
        a_, b_, c_ = errs[k - 1], errs[k], errs[k + 1]; den = a_ - 2 * b_ + c_
        return k - 6 + (0.5 * (a_ - c_) / den if den > 1e-6 else 0)
    return float(k - 6)

# per row: the run(s) to fill, its source side and the shifts at its two seams (estimated every 8 rows, smoothed)
info = {}
for y in range(0, H, 8):
    for x0, x1 in runs(fill[y]):
        w = x1 - x0 + 1
        okL = x0 - SEAM - w - 2 >= 0 and mesh[y, max(x0 - w - 2, 0):x0 - 2].mean() > 0.85
        okR = x1 + w + 2 < W and mesh[y, x1 + 3:x1 + w + 3].mean() > 0.85
        if okL:     # source = mesh left of the run, copied right by w+2
            dL = shift(y, np.arange(x0 - SEAM, x0), np.arange(x0 - SEAM, x0) - w - 2)
            dR = shift(y, np.arange(x1 + 1, min(x1 + 1 + SEAM, W)), np.arange(x0 - 2 - SEAM, x0 - 2)[:min(SEAM, W - x1 - 1)]) if okR else dL
            info.setdefault('L', []).append((y, x0, x1, dL, dR))
        elif okR:
            dR = shift(y, np.arange(x1 + 1, x1 + 1 + SEAM), np.arange(x1 + 1, x1 + 1 + SEAM) + w + 2)
            info.setdefault('R', []).append((y, x0, x1, dR, dR))

def smooth_lookup(entries):
    ys = np.array([e[0] for e in entries], float); dl = np.array([e[3] for e in entries]); dr = np.array([e[4] for e in entries])
    k = np.exp(-((ys[:, None] - ys[None, :]) / 24.0) ** 2); k /= k.sum(1, keepdims=True)
    return ys, k @ dl, k @ dr

tabs = {side: smooth_lookup(e) for side, e in info.items()}
for y in range(H):
    for x0, x1 in runs(fill[y]):
        w = x1 - x0 + 1; xs = np.arange(x0, x1 + 1); t = (xs - x0) / max(w - 1, 1)
        okL = x0 - w - 2 >= 0 and mesh[y, max(x0 - w - 2, 0):x0 - 2].mean() > 0.85
        okR = x1 + w + 2 < W and mesh[y, x1 + 3:x1 + w + 3].mean() > 0.85
        side = 'L' if okL and 'L' in tabs else ('R' if okR and 'R' in tabs else None)
        if side is None: continue
        ys_, dl, dr = tabs[side]; j = np.argmin(np.abs(ys_ - y)); dL, dR = dl[j], dr[j]
        d = dL + (dR - dL) * (t * t * (3 - 2 * t))
        sx = (xs - w - 2) if side == 'L' else (xs + w + 2)
        mapx = sx.astype(np.float32)[None, :]; mapy = (y + d).astype(np.float32)[None, :]
        out[y, xs] = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)[0]
edge = cv2.dilate(fill.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~cv2.erode(fill.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
bl = cv2.GaussianBlur(out, (0, 0), 1.0); out[edge] = bl[edge]
cv2.imwrite(dst, np.clip(out, 0, 255).astype(np.uint8))
if dbg:
    d = img.copy(); d[fill] = d[fill] * 0.4 + np.array([0, 0, 255]) * 0.6
    cv2.imwrite(dbg, cv2.resize(np.hstack([d, out]).astype(np.uint8), None, fx=0.35, fy=0.35))
print('ok', dst, 'filled px', int(fill.sum()))
