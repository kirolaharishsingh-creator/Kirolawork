# Replace parts of the cleaned chair model (chair_m8c_cleaned.blend) with separately generated Tripo part models.
# Each Tripo GLB is decimated, then fitted onto the old part it replaces: the best of the 24 axis rotations (plus a
# mirror where allowed) after a bounding-box scale/centre match, refined by similarity ICP (rotation, uniform scale,
# translation) on nearest neighbours. The fitted mesh takes the old object's name and materials slot order is kept
# from Tripo (its own PBR textures). Writes OUT blend + a JSON report of fit errors.
# Usage: python fit_tripo_parts.py   (env: OLD, OUT, PARTS='name=glb;...', FACES target per part)
import bpy, bmesh, os, sys, json, math, itertools, numpy as np, mathutils
from scipy.spatial import cKDTree

OLD = os.environ.get('OLD', '/home/user/Kirolawork/exploded/3d/models/chair_m8c_cleaned.blend')
OUT = os.environ['OUT']; FACES = int(os.environ.get('FACES', 250000))
PARTS = [p.split('=') for p in os.environ['PARTS'].split(';')]      # objname=path.glb (arm_r and arm_l may share a file)
MIRROR_OK = {'arm_l', 'arm_r'}

bpy.ops.wm.open_mainfile(filepath=OLD); sc = bpy.context.scene
for p in [p for p in os.environ.get('KEEP_ONLY', '').split(',') if p]: pass
def verts(o):
    return np.array([o.matrix_world @ v.co for v in o.data.vertices], np.float64)

rots = []
for perm in itertools.permutations(range(3)):
    for signs in itertools.product((1, -1), repeat=3):
        M = np.zeros((3, 3))
        for i, (p, s) in enumerate(zip(perm, signs)): M[i, p] = s
        rots.append(M)                                   # 48: proper (det +1) and mirrored (det -1)

def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0); A = src - ms; B = dst - md
    U, S, Vt = np.linalg.svd(B.T @ A / len(src)); d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d]); R = U @ D @ Vt; s = np.trace(np.diag(S) @ D) / (A ** 2).sum(1).mean()
    return s, R, md - s * R @ ms

report = {}
for name, path in PARTS:
    names = name.split('+'); olds = [bpy.data.objects[n] for n in names]; name = names[0]; old = olds[0]
    tgt = np.vstack([verts(o) for o in olds])            # e.g. base+wheel_0+...: the Tripo base includes its castors
    tree_t = cKDTree(tgt[np.random.default_rng(0).choice(len(tgt), min(20000, len(tgt)), replace=False)])
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before and o.type == 'MESH'][0]
    for o in [o for o in bpy.data.objects if o not in before and o is not new]: bpy.data.objects.remove(o, do_unlink=True)
    new.parent = None; mw = new.matrix_world.copy(); new.data.transform(mw); new.matrix_world.identity()
    if len(new.data.polygons) > FACES:                    # decimate (collapse) to a workable size
        m = new.modifiers.new('dec', 'DECIMATE'); m.ratio = FACES / len(new.data.polygons)
        bpy.context.view_layer.objects.active = new; bpy.ops.object.modifier_apply(modifier='dec')
    src_all = verts(new); src = src_all[np.random.default_rng(1).choice(len(src_all), min(20000, len(src_all)), replace=False)]
    best = None
    for M in rots:
        if np.linalg.det(M) < 0 and name not in MIRROR_OK: continue
        if M[2, 2] != 1: continue                         # Tripo GLBs are Z-up: never flip a part upside down (seat!)
        p = src @ M.T; lo, hi = p.min(0), p.max(0); tlo, thi = tgt.min(0), tgt.max(0)
        s = np.median((thi - tlo) / np.maximum(hi - lo, 1e-6)); t = (tlo + thi) / 2 - s * (lo + hi) / 2
        q = s * p + t; d = tree_t.query(q)[0].mean() + cKDTree(q).query(tree_t.data)[0].mean()
        if best is None or d < best[0]: best = (d, M, s, t)
    d0, M, s, t = best; T = np.eye(4); T[:3, :3] = s * M; T[:3, 3] = t
    cur = src @ T[:3, :3].T + T[:3, 3]
    for it in range(40):                                  # similarity ICP, symmetric correspondences
        _, i1 = tree_t.query(cur); tq = cKDTree(cur); _, i2 = tq.query(tree_t.data)
        A = np.vstack([cur, cur[i2]]); B = np.vstack([tree_t.data[i1], tree_t.data])
        si, Ri, ti = umeyama(A, B); step = np.eye(4); step[:3, :3] = si * Ri; step[:3, 3] = ti
        T = step @ T; cur = src @ T[:3, :3].T + T[:3, 3]
    err = tree_t.query(cur)[0].mean() + cKDTree(cur).query(tree_t.data)[0].mean()
    new.data.transform(mathutils.Matrix(T.tolist()))
    if np.linalg.det(T[:3, :3]) < 0: new.data.flip_normals()
    new.data.update()
    for o in olds: bpy.data.objects.remove(o, do_unlink=True); new.name = name; new.data.name = name
    report[name] = {'init_err_mm': round(d0 * 1000, 1), 'final_err_mm': round(err * 1000, 1), 'scale': round(float(np.cbrt(abs(np.linalg.det(T[:3, :3])))), 4),
                    'mirrored': bool(np.linalg.det(T[:3, :3]) < 0), 'faces': len(new.data.polygons)}
    print('FIT', name, report[name], flush=True)
bpy.ops.wm.save_as_mainfile(filepath=OUT)
json.dump(report, open(OUT.replace('.blend', '_fit.json'), 'w'), indent=1)
