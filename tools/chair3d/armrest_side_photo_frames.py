# Armrest 4th movement (pad slides sideways) from two real portrait photos: start + end frames for Kling, 16:9.
#  1. align the end photo to the start on the static parts (backrest, lumbar, seat; armrest zone masked out),
#     ORB + RANSAC homography refined by ECC, so only the pad moves between the frames
#  2. crop a 16:9 band (y0..y0+H) and extend the backdrop on the right: each row continues with the median of the
#     photo's right-edge strip (keeps the wall-to-floor gradient), feathered into the photo, light grain
# Usage: python armrest_side_photo_frames.py start.jpg end.jpg outdir   (env: Y0 520, H 1100, EXCL x0,y0,x1,y1)
import cv2, numpy as np, sys, os
a_path, b_path, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
Y0, H = int(os.environ.get('Y0', 520)), int(os.environ.get('H', 1100))
EX = [int(v) for v in os.environ.get('EXCL', '380,720,820,1700').split(',')]
A, B = cv2.imread(a_path), cv2.imread(b_path)

def feats(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); m = np.zeros(g.shape, np.uint8)
    m[:, :int(im.shape[1] * 0.75)] = 255; m[EX[1]:EX[3], EX[0]:EX[2]] = 0   # chair only, armrest out
    return cv2.ORB_create(6000).detectAndCompute(g, m)
k1, d1 = feats(A); k2, d2 = feats(B)
mt = cv2.BFMatcher(cv2.NORM_HAMMING, True).match(d1, d2)
p1 = np.float32([k1[x.queryIdx].pt for x in mt]); p2 = np.float32([k2[x.trainIdx].pt for x in mt])
Hm, inl = cv2.findHomography(p2, p1, cv2.RANSAC, 2.5); print('inliers', int(inl.sum()), '/', len(mt), 'shift', Hm[:2, 2].round(1))
# ECC refinement on the static band (backrest, lumbar, seat edge), armrest zone out
g1, g2 = [cv2.GaussianBlur(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), (5, 5), 0) for x in (A, B)]
em = np.zeros(g1.shape, np.uint8); em[Y0:Y0 + H, :int(A.shape[1] * 0.75)] = 255; em[EX[1]:EX[3], EX[0]:EX[2]] = 0
W = np.linalg.inv(Hm).astype(np.float32)
_, W = cv2.findTransformECC(g1, g2, W, cv2.MOTION_HOMOGRAPHY, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-7), em, 5)
B = cv2.warpPerspective(B, W, (A.shape[1], A.shape[0]), flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)

W = int(round(H * 16 / 9)); rng = np.random.default_rng(1)
def widen(im):
    band = im[Y0:Y0 + H].astype(np.float32); w0 = band.shape[1]
    edge = np.median(band[:, w0 - 120:w0 - 10], axis=1)                     # per-row backdrop colour
    edge = cv2.GaussianBlur(edge[:, None, :], (1, 41), 0)[:, 0, :]
    ext = np.repeat(edge[:, None, :], W, axis=1)
    ext += rng.normal(0, 1.2, ext.shape).astype(np.float32)                 # match photo grain
    a = np.clip((np.arange(w0) - (w0 - 160)) / 150.0, 0, 1)[None, :, None]  # feather the last 150 px into the fill
    ext[:, :w0] = band * (1 - a) + ext[:, :w0] * a
    return cv2.resize(np.clip(ext, 0, 255).astype(np.uint8), (1920, 1080), interpolation=cv2.INTER_AREA)
s, e = widen(A), widen(B)
cv2.imwrite(out + '/armside_start.png', s); cv2.imwrite(out + '/armside_end.png', e)
d = cv2.absdiff(s, e).max(2); print('static-zone diff (mean, outside arm):', round(float(d[:, :int(380 * 1080 / H)].mean()), 2))
sbs = cv2.hconcat([cv2.resize(s, (960, 540)), cv2.resize(e, (960, 540))])
cv2.imwrite(out + '/armside_pair.jpg', sbs, [cv2.IMWRITE_JPEG_QUALITY, 92])
