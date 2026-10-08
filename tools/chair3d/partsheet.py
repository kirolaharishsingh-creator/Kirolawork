# Lay the per-part renders out in a labelled grid on the studio grey.
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
def over(rgba):
    a8 = rgba[..., 3]; im = rgba.astype(np.float32) / 255; a = im[..., 3:]
    ac = np.asarray(Image.fromarray(a8).filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.MinFilter(9)), np.float32)[..., None] / 255
    ac = np.maximum(ac, a); bg = np.array([202, 198, 194], np.float32) / 255; fill = np.array([40, 40, 42], np.float32) / 255
    return Image.fromarray((np.clip(im[..., :3] * a + fill * (ac - a) + bg * (1 - ac), 0, 1) * 255).astype(np.uint8))
order = ['headrest', 'backrest', 'frame', 'lumbar', 'seat', 'arm_l', 'arm_r', 'mechanism', 'gas_lift', 'base', 'wheel0']
label = {'headrest': 'Headrest', 'backrest': 'Backrest', 'frame': 'Spine frame', 'lumbar': 'Lumbar', 'seat': 'Seat',
         'arm_l': 'Armrest L', 'arm_r': 'Armrest R', 'mechanism': 'Mechanism', 'gas_lift': 'Gas lift', 'base': 'Base', 'wheel0': 'Wheel'}
T = 300; cols = 4; rows = (len(order) + 1) // 2
sheet = Image.new('RGB', (cols * T, rows * (T + 24)), (202, 198, 194)); d = ImageDraw.Draw(sheet)
for i, n in enumerate(order):
    for j, s in enumerate('ab'):
        k = i * 2 + j; x, y = (k % cols) * T, (k // cols) * (T + 24)
        sheet.paste(over(np.asarray(Image.open(f'parts/{n}_{s}.png').convert('RGBA'))), (x, y + 24)); d.text((x + 8, y + 6), f'{label[n]} ({"back" if s == "a" else "front"})', fill=(30, 30, 30))
sheet.save('partsheet.jpg', quality=82)
