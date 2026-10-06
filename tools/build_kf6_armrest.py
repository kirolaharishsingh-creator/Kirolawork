import cv2, numpy as np
from PIL import Image, ImageOps
im = np.array(ImageOps.exif_transpose(Image.open('chair_photos/12_usp_armrest_4D_positions/34.jpg')).convert('RGB')).astype(np.float32)
Y0 = 1650
crop = im[Y0:Y0+1899]
crop = cv2.resize(crop, (3840, 2160), interpolation=cv2.INTER_LANCZOS4)
lum = crop.mean(2)
a = np.clip((185 - lum) / 45, 0, 1)
a = cv2.GaussianBlur(a, (0, 0), 1.5)[..., None]
H, W = 2160, 3840
yy, xx = np.mgrid[0:H, 0:W]
d = np.sqrt(((xx - W*0.45)/W)**2 + ((yy - H*0.4)/H)**2)
bg = np.clip(236 - 40*d, 185, 236)[..., None] * np.array([1, 0.985, 0.97])
out = crop * a + bg * (1 - a)
Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save('real_photo_keyframes/kf6_armrest_4k.jpg', quality=95)
Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).resize((1280, 720)).save('/tmp/claude-0/-home-user-Kirolawork/2e913183-d11c-511c-b00f-6ae556cd9297/scratchpad/kf6_prev.jpg')
