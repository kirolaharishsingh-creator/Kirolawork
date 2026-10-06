"""Place a white-background chair cutout into a 4K light-grey studio.
Usage: python3 tools/build_keyframe.py <cutout.jpg> <out.jpg>
All frames use the same placement so start/end frames line up."""
import sys, numpy as np, cv2
from PIL import Image, ImageFilter

W, H = 3840, 2160
REF_TOP, REF_BOTTOM, REF_LEFT, REF_W = 370, 1672, 200, 715   # framing taken from photo 15
K = int(H * 0.80) / (REF_BOTTOM - REF_TOP)
X0 = (W - REF_W * K) / 2 - REF_LEFT * K
Y0 = int(H * 0.92) - int(H * 0.80) - REF_TOP * K

def cutout(path, clear_trapped=True):
    src = np.array(Image.open(path).convert('RGB')).astype(np.float32)
    mn = src.min(-1); h, w = mn.shape
    ff = (mn > 238).astype(np.uint8); m = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, m, (0, 0), 2); outer = ff == 2
    holes = np.zeros((h, w), bool)
    # white background trapped inside the base (between chrome legs, lever loops)
    if clear_trapped:
        trapped = (ff == 1).astype(np.uint8); trapped[:int(h * 0.6)] = 0
        n, lab, st, _ = cv2.connectedComponentsWithStats(trapped)
        for i in range(1, n):
            if st[i, 4] > 60: holes |= lab == i
        outer = outer | holes
    alpha = np.where(outer, 0, 1).astype(np.float32)
    inner = np.clip((mn - 215) / 30, 0, 1); inner[int(h * 0.575):] = 0      # see-through mesh, not chrome
    alpha = np.minimum(alpha, 1 - inner * (~outer))
    alpha = cv2.erode(alpha, np.ones((3, 3), np.uint8)); alpha = cv2.GaussianBlur(alpha, (0, 0), 0.8)
    edge = (alpha > 0) & (cv2.erode((alpha > 0.8).astype(np.uint8), np.ones((5, 5), np.uint8)) == 0)
    near_hole = cv2.dilate(holes.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    fix = edge & (src.mean(-1) > 150) & ~near_hole; src[fix] *= 0.35       # remove white fringe
    # around cleared holes the edge is bright chrome: no darkening, just a softer, tighter alpha
    soft = cv2.GaussianBlur(cv2.erode(alpha, np.ones((3, 3), np.uint8)), (0, 0), 1.2)
    alpha = np.where(near_hole, soft, alpha)
    return src, alpha

def wheels(src, alpha):
    ys = np.where(alpha.max(1) > 0.5)[0]; top, bot = ys.min(), ys.max()
    low = (alpha > 0.5) & (src.max(-1) < 90); low[:int(top + 0.80 * (bot - top))] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(low.astype(np.uint8))
    out = []
    for x, y, ww, hh, ar in st[1:]:
        if ar > 600 and 0.6 < hh / max(ww, 1) < 2.5 and ww > 40:
            out.append((x + ww / 2, y + hh, ww))
    return out

def rim(src, alpha):
    # rim light: brighten the outer edge band of the chair
    band = cv2.GaussianBlur(alpha, (0, 0), 1.5) - cv2.GaussianBlur(cv2.erode(alpha, np.ones((9, 9), np.uint8)), (0, 0), 1.5)
    band = np.clip(band, 0, 1)[..., None]; return np.clip(src + band * 55, 0, 255)

def build(path, outp):
    src, alpha = cutout(path)
    h, w = alpha.shape
    ps, pa = cutout(path, clear_trapped=False); wh = wheels(rim(ps, pa), pa)  # wheels from the plain cutout
    src = rim(src, alpha)
    rgba = Image.fromarray(np.dstack([src, alpha * 255]).astype(np.uint8), 'RGBA')
    chair = rgba.resize((int(w * K), int(h * K)), Image.LANCZOS)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - W / 2) / (W * 0.55)) ** 2 + ((yy - H * 0.42) / (H * 0.75)) ** 2)
    base = np.clip(236 - 40 * d, 185, 236)
    floor = np.clip((yy - H * 0.70) / (H * 0.30), 0, 1)                     # slightly darker floor
    base = base - floor * 14
    bg = Image.fromarray(np.stack([base, base * 0.985, base * 0.97], -1).astype(np.uint8)).convert('RGBA')
    pts = [(X0 + x * K, Y0 + y * K, ww * K) for x, y, ww in wh]
    Y, X = np.ogrid[0:H, 0:W]
    if pts:
        px = np.array([p[0] for p in pts]); py = np.array([p[1] for p in pts])
        rx = (px.max() - px.min()) * 0.62 + 1; ry = (py.max() - py.min()) * 0.75 + 30
        e = ((X - px.mean()) / rx) ** 2 + ((Y - py.mean()) / ry) ** 2
        sa = np.where(e < 1, (1 - e) * 60, 0)
        bg.alpha_composite(Image.fromarray(np.dstack([np.zeros((H, W, 3), np.uint8), sa.astype(np.uint8)])).filter(ImageFilter.GaussianBlur(40)))
        cs = np.zeros((H, W), np.float32)
        for x, y, ww in pts:
            e = ((X - x) / (ww * 0.75)) ** 2 + ((Y - (y - 4)) / (ww * 0.16)) ** 2
            cs += np.where(e < 1, (1 - e) * 190, 0)
        bg.alpha_composite(Image.fromarray(np.dstack([np.zeros((H, W, 3), np.uint8), np.clip(cs, 0, 230).astype(np.uint8)])).filter(ImageFilter.GaussianBlur(7)))
    bg.alpha_composite(chair, (int(X0), int(Y0)))
    bg.convert('RGB').save(outp, quality=94)
    print(outp, 'wheels found:', len(pts))

if __name__ == '__main__':
    build(sys.argv[1], sys.argv[2])
