import cv2, numpy as np
from PIL import Image, ImageOps
im = np.array(ImageOps.exif_transpose(Image.open('chair_photos/14_usp_underside_mechanism/20.jpg')).convert('RGB')).astype(np.float32)
crop = cv2.resize(im, (3840, 2160), interpolation=cv2.INTER_AREA)
H, W = 2160, 3840
lum = crop.mean(2)
a = np.clip((190 - lum) / 50, 0, 1)
# protect the chrome base at the bottom centre
yy, xx = np.mgrid[0:H, 0:W]
chrome = ((yy > H*0.93) & (xx > W*0.40) & (xx < W*0.66)).astype(np.float32)
chrome = cv2.GaussianBlur(chrome, (0, 0), 30)[..., None]
a = cv2.GaussianBlur(a, (0, 0), 1.5)[..., None]
d = np.sqrt(((xx - W*0.5)/W)**2 + ((yy - H*0.35)/H)**2)
bg = np.clip(236 - 40*d, 185, 236)[..., None] * np.array([1, 0.985, 0.97])
out = crop * a + bg * (1 - a)
# chrome zone: keep the photo, only shift the wall tone to the studio grey
wall = (chrome[..., 0] > 0.5) & (a[..., 0] < 0.02)
shifted = crop + (1 - a) * (bg - crop[wall].mean(0))
out = out * (1 - chrome) + shifted * chrome
out = np.clip(out, 0, 255).astype(np.uint8)
Image.fromarray(out).save('real_photo_keyframes/kf7_mechanism_4k.jpg', quality=95)
