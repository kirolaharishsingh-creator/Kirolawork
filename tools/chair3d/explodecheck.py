# Point-cloud preview of the exploded hold pose, coloured by part, before and after the connectivity cleanup.
import numpy as np
from PIL import Image
from parts import label_components, PART_NAMES
co = np.load('co.npy'); lab = np.load('lab.npy'); tri = np.load('tri.npy')
OFF = {'headrest': (-0.05, 0, 0.22), 'backrest': (-0.10, 0, 0.12), 'frame': (-0.30, 0, 0.02), 'lumbar': (-0.15, 0, 0),
       'seat': (0, 0, 0.14), 'arm_r': (0, 0.17, 0.06), 'arm_l': (0, -0.17, 0.06), 'mechanism': (0, 0, -0.04),
       'gas_lift': (0, 0, -0.10), 'base': (0, 0, -0.18)}
pal = np.array([(230,60,60),(60,160,230),(250,170,30),(60,200,90),(150,90,220),(240,90,200),(200,60,140),
                (120,120,120),(30,200,200),(90,90,40),(20,20,20),(70,40,10),(120,60,0),(0,70,120),(0,120,60)], np.uint8)
def view(part, az, el=12, size=600):
    off = np.array([OFF.get(n, (0, 0, -0.26)) for n in PART_NAMES])[part]
    p0 = co + off
    a, e = np.radians(az), np.radians(el)
    R = np.array([[-np.sin(a), np.cos(a), 0], [-np.sin(e)*np.cos(a), -np.sin(e)*np.sin(a), np.cos(e)], [np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)]])
    p = p0 @ R.T
    u = ((p[:, 0] / 1.7 + 0.5) * size).astype(int); v = ((0.5 - p[:, 1] / 1.7) * size).astype(int)
    o = np.argsort(p[:, 2]); img = np.full((size, size, 3), 255, np.uint8)
    ok = (u >= 0) & (u < size) & (v >= 0) & (v < size); o = o[ok[o]]; img[v[o], u[o]] = pal[part[o]]
    return img
before = label_components(co, lab)[lab]
after = label_components(co, lab, tri)[lab]
print('changed verts', int((before != after).sum()))
Image.fromarray(np.vstack([np.hstack([view(before, 235), view(before, 270)]), np.hstack([view(after, 235), view(after, 270)])])).resize((800, 800)).save('explodecheck.jpg', quality=70)
