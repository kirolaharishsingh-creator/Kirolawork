# Lay the per-part renders out in labelled grids on the studio grey (full resolution, split into pages).
import sys, numpy as np
from PIL import Image, ImageDraw
from comp import over as _over
BGc = np.array([202, 198, 194], np.float32) / 255
order = ['headrest', 'backrest', 'frame', 'lumbar', 'seat', 'arm_l', 'arm_r', 'mechanism', 'gas_lift', 'base', 'wheel0']
label = {'headrest': 'Headrest', 'backrest': 'Backrest', 'frame': 'Spine frame', 'lumbar': 'Lumbar', 'seat': 'Seat',
         'arm_l': 'Armrest L', 'arm_r': 'Armrest R', 'mechanism': 'Mechanism', 'gas_lift': 'Gas lift', 'base': 'Base', 'wheel0': 'Wheel'}
per_page = int(sys.argv[1]) if len(sys.argv) > 1 else 4           # parts per page (2 views each)
T = Image.open(f'parts/{order[0]}_a.png').size[0]; cols = 4
for p in range(0, len(order), per_page):
    names = order[p:p + per_page]; rows = (len(names) * 2 + cols - 1) // cols
    sheet = Image.new('RGB', (cols * T, rows * (T + 30)), (202, 198, 194)); d = ImageDraw.Draw(sheet)
    for i, n in enumerate(names):
        for j, s in enumerate('ab'):
            k = i * 2 + j; x, y = (k % cols) * T, (k // cols) * (T + 30)
            sheet.paste(Image.fromarray(_over(np.asarray(Image.open(f'parts/{n}_{s}.png').convert('RGBA')), BGc)), (x, y + 30))
            d.text((x + 10, y + 8), f'{label[n]} ({"back" if s == "a" else "front"})', fill=(30, 30, 30))
    sheet.save(f'partsheet_{p // per_page + 1}.jpg', quality=80)
    print('page', p // per_page + 1, names)
