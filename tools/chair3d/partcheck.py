# Point-cloud views coloured by part, to check the split (no Blender needed).
import numpy as np
from PIL import Image
from parts import label_components, PART_NAMES
co = np.load('co.npy'); lab = np.load('lab.npy')
part = label_components(co, lab)[lab]
pal = np.array([(230,60,60),(60,160,230),(250,170,30),(60,200,90),(150,90,220),(240,90,200),(200,60,140),
                (120,120,120),(30,200,200),(90,90,40),(20,20,20),(70,40,10),(120,60,0),(0,70,120),(0,120,60)], np.uint8)
def view(az, el, size=700):
    a, e = np.radians(az), np.radians(el)
    R = np.array([[-np.sin(a), np.cos(a), 0], [-np.sin(e)*np.cos(a), -np.sin(e)*np.sin(a), np.cos(e)], [np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)]])
    p = co @ R.T
    u = ((p[:, 0] / 1.1 + 0.5) * size).astype(int); v = ((0.5 - p[:, 1] / 1.1) * size).astype(int)
    o = np.argsort(p[:, 2])                      # far first, near drawn last
    img = np.full((size, size, 3), 255, np.uint8)
    ok = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    o = o[ok[o]]; img[v[o], u[o]] = pal[part[o]]
    return img
tiles = [view(270, 0), view(0, 0), view(180, 0), view(215, 20)]   # side(from -y), front, back, rear 3/4
Image.fromarray(np.hstack(tiles)).save('partcheck.jpg', quality=70)
print({n: int((part == i).sum()) for i, n in enumerate(PART_NAMES)})
