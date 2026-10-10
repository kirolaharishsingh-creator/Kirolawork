# End frame for the assembly video: keyframe kf1B (real photo 31 on the studio backdrop, 1080p framing the 3D render
# was fitted to) with its chrome legs replaced. kf1B's cut-out made the chrome legs semi-transparent; photo_to_studio
# with CHROME=0.18 keeps them solid, so the legs (hand-traced polygons) are taken from that version, placed by an ECC fit.
# Usage: python3 photo31_endframe.py kf1B_1080.png p31_c018.png out.png      (p31_c018: photo_to_studio W=2400 CHROME=0.18)
import sys, numpy as np, cv2

kf, c18, out = sys.argv[1:4]
k = cv2.imread(kf).astype(np.float32); src = cv2.imread(c18).astype(np.float32)
LEGS = [  # in the W=2400 photo_to_studio image: front-left leg, front-right leg, front leg + castor post
    [(1112, 2998), (1128, 3052), (528, 3232), (496, 3196)],
    [(1250, 2990), (1268, 2985), (1872, 3196), (1852, 3232), (1240, 3050)],
    [(1150, 2950), (1222, 2950), (1218, 3372), (1158, 3372)]]
ms = (src.mean(2) < 100).astype(np.float32); mk = (k.mean(2) < 100).astype(np.float32)
ys, xs = np.where(ms > 0); yt, xt = np.where(mk > 0); s = (yt.max() - yt.min()) / (ys.max() - ys.min())
A = np.float32([[s, 0, xt.min() - s * xs.min()], [0, s, yt.min() - s * ys.min()]])
W2 = np.eye(2, 3, dtype=np.float32)
_, W2 = cv2.findTransformECC(cv2.GaussianBlur(mk, (0, 0), 2), cv2.GaussianBlur(cv2.warpAffine(ms, A, (1920, 1080)), (0, 0), 2),
                             W2, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-7), None, 5)
T = (np.linalg.inv(np.vstack([W2, [0, 0, 1]])) @ np.vstack([A, [0, 0, 1]]))[:2]
m = np.zeros(src.shape[:2], np.float32)
for P in LEGS: cv2.fillPoly(m, [np.int32(P)], 1.0)
ph = cv2.warpAffine(src, T, (1920, 1080), flags=cv2.INTER_AREA)
mm = cv2.GaussianBlur(cv2.warpAffine(m, T, (1920, 1080), flags=cv2.INTER_AREA), (0, 0), 0.8)[..., None]
res = k * (1 - mm) + ph * mm
cv2.imwrite(out, np.clip(res, 0, 255).astype(np.uint8)); print('ok', out)
