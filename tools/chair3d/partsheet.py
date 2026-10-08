# Lay the per-part renders out in a labelled grid on the studio grey.
from PIL import Image, ImageDraw
order = ['headrest', 'backrest', 'frame', 'lumbar', 'seat', 'arm_l', 'arm_r', 'mechanism', 'gas_lift', 'base', 'wheel0']
label = {'headrest': 'Headrest', 'backrest': 'Backrest', 'frame': 'Spine frame', 'lumbar': 'Lumbar', 'seat': 'Seat',
         'arm_l': 'Armrest L', 'arm_r': 'Armrest R', 'mechanism': 'Mechanism', 'gas_lift': 'Gas lift', 'base': 'Base', 'wheel0': 'Wheel'}
T = 300; cols = 4; rows = (len(order) + 1) // 2
sheet = Image.new('RGB', (cols * T, rows * (T + 24)), (202, 198, 194)); d = ImageDraw.Draw(sheet)
for i, n in enumerate(order):
    for j, s in enumerate('ab'):
        k = i * 2 + j; x, y = (k % cols) * T, (k // cols) * (T + 24)
        im = Image.open(f'parts/{n}_{s}.png').convert('RGBA'); bg = Image.new('RGBA', im.size, (202, 198, 194, 255)); bg.alpha_composite(im)
        sheet.paste(bg.convert('RGB'), (x, y + 24)); d.text((x + 8, y + 6), f'{label[n]} ({"back" if s == "a" else "front"})', fill=(30, 30, 30))
sheet.save('partsheet.jpg', quality=82)
