# Backrest (back, front) next to the lumbar, large, to compare the fabric.
import numpy as np
from PIL import Image, ImageDraw
from comp import over
BG = np.array([202, 198, 194], np.float32) / 255
tiles = [('Backrest (back)', 'parts/backrest_a.png'), ('Backrest (front)', 'parts/backrest_b.png'), ('Lumbar (front)', 'parts/lumbar_b.png')]
ims = [Image.fromarray(over(np.asarray(Image.open(p).convert('RGBA')), BG)) for _, p in tiles]
s = ims[0].size[0]; sheet = Image.new('RGB', (s * 3, s + 24), (202, 198, 194)); d = ImageDraw.Draw(sheet)
for i, ((t, _), im) in enumerate(zip(tiles, ims)):
    sheet.paste(im, (i * s, 24)); d.text((i * s + 8, 6), t, fill=(30, 30, 30))
sheet.save('fabric.jpg', quality=85)
