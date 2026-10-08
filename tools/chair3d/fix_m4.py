# Prepare the clean-front Tripo chair (15 parts, textured) for the exploded animation.
# Usage: python3 fix_m4.py model.glb bf.png bb.png out.blend   (env HEAD_PHOTO=headrest_front_photo15_hires.png)
#   - backrest: the mesh inside the frame ring gets the real fabric photos (Tripo baked the studio
#     background into it as white blotches); its small dents are smoothed onto a fitted surface
#   - base: the gas-lift column and the fifth caster are split off; the base becomes matte black plastic
#   - parts are renamed to the animation's names (same layout as tripo_fix.py's output, for anim2.py)
# Model axes (as exported by Tripo): z up, chair front towards -y, x across.
import bpy, bmesh, sys, os, numpy as np
from scipy import ndimage

GLB, FRONT, BACK, OUT = sys.argv[-4:]
GROUPS = {'headrest': [9], 'backrest': [0], 'frame': [6, 3], 'lumbar': [2, 10], 'seat': [5],
          'arm_l': [7], 'arm_r': [8], 'mechanism': [4], 'gas_lift': [], 'base': [1], 'wheels': [11, 12, 13, 14]}

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=GLB)
P = {int(o.name.split('_')[-1].split('.')[0]): o for o in bpy.data.objects if o.type == 'MESH'}
for o in P.values():
    o.data.transform(o.matrix_world); o.matrix_world.identity(); o.parent = None

def verts(o):
    v = np.zeros(len(o.data.vertices) * 3); o.data.vertices.foreach_get('co', v); return v.reshape(-1, 3)

def face_centres(o):
    c = np.zeros(len(o.data.polygons) * 3); o.data.polygons.foreach_get('center', c); return c.reshape(-1, 3)

def face_normals(o):
    n = np.zeros(len(o.data.polygons) * 3); o.data.polygons.foreach_get('normal', n); return n.reshape(-1, 3)

def split_faces(o, sel, name):
    new = o.copy(); new.data = o.data.copy(); new.name = name; bpy.context.scene.collection.objects.link(new)
    for ob, keep in ((o, ~sel), (new, sel)):
        bm = bmesh.new(); bm.from_mesh(ob.data); bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[f for f, k in zip(bm.faces, keep) if not k], context='FACES')
        bm.to_mesh(ob.data); bm.free()
    return new

# ---- base: gas-lift column (above the hub, on the axis) and the fifth caster come off
b = P[1]; v = verts(b)
top = v[v[:, 2] > 0.2]; ax = (top[:, :2].min(0) + top[:, :2].max(0)) / 2
HZ = float(os.environ.get('HUB_TOP', 0.16))
bm = bmesh.new(); bm.from_mesh(b.data)               # a clean flat cut where the column leaves the hub
bmesh.ops.bisect_plane(bm, geom=list(bm.verts) + list(bm.edges) + list(bm.faces), plane_co=(0, 0, HZ), plane_no=(0, 0, 1))
bm.to_mesh(b.data); bm.free(); b.data.update()
c = face_centres(b); r = np.hypot(c[:, 0] - ax[0], c[:, 1] - ax[1])
col = (c[:, 2] > HZ) & (r < 0.045)
P[90] = split_faces(b, col, 'gas_column'); GROUPS['gas_lift'].append(90)
for o in (b, P[90]):                                  # close both cut openings
    bm = bmesh.new(); bm.from_mesh(o.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    edges = [e for e in bm.edges if e.is_boundary and all(abs(vv.co.z - HZ) < 1e-4 for vv in e.verts)]
    if edges: bmesh.ops.holes_fill(bm, edges=edges, sides=0)
    bm.to_mesh(o.data); bm.free(); o.data.update()
c = face_centres(b); low = c[:, 2] < 0.06
wxy = c[low, :2].mean(0); wsel = low & (np.hypot(c[:, 0] - wxy[0], c[:, 1] - wxy[1]) < 0.045)
P[91] = split_faces(b, wsel, 'wheel_5'); GROUPS['wheels'].append(91)
print('base: column', int(col.sum()), 'faces, caster', int(wsel.sum()), 'faces at', wxy.round(3))

# ---- materials
def photo_box(img, inset=0.09):
    w, h = img.size; px = np.array(img.pixels[:], np.float32).reshape(h, w, 4)
    ys, xs = np.nonzero(px[..., :3].mean(-1) < 0.6)
    u0, u1, v0, v1 = xs.min() / w, xs.max() / w, ys.min() / h, ys.max() / h
    du, dv = (u1 - u0) * inset, (v1 - v0) * inset
    return u0 + du, u1 - du, v0 + dv, v1 - dv

def fabric(name, path, uvname):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; pb = nt.nodes['Principled BSDF']
    pb.inputs['Roughness'].default_value = 0.7; pb.inputs['Specular IOR Level'].default_value = 0.25
    t = nt.nodes.new('ShaderNodeTexImage'); t.image = bpy.data.images.load(path); t.extension = 'EXTEND'
    u = nt.nodes.new('ShaderNodeUVMap'); u.uv_map = uvname; nt.links.new(u.outputs[0], t.inputs['Vector'])
    gm = nt.nodes.new('ShaderNodeGamma'); gm.inputs[1].default_value = float(os.environ.get('FAB_GAMMA', 1.25))
    nt.links.new(t.outputs['Color'], gm.inputs[0]); nt.links.new(gm.outputs[0], pb.inputs['Base Color'])
    return m, photo_box(t.image)

fab_f, box_f = fabric('backrest_fabric', FRONT, 'front_uv')
fab_b, box_b = fabric('backrest_back_fabric', BACK, 'back_uv')

# ---- backrest: frame ring stays, mesh inside it gets the real fabric
bk = P[0]; v = verts(bk); STEP = 0.0015; PAD = 20
x0, z0 = v[:, 0].min() - PAD * STEP, v[:, 2].min() - PAD * STEP
nx, nz = int(np.ptp(v[:, 0]) / STEP) + 2 * PAD + 1, int(np.ptp(v[:, 2]) / STEP) + 2 * PAD + 1
sil = np.zeros((nx, nz), bool); sil[((v[:, 0] - x0) / STEP).astype(int), ((v[:, 2] - z0) / STEP).astype(int)] = True
sil = ndimage.binary_fill_holes(ndimage.binary_closing(sil, iterations=4))
inner = ndimage.binary_erosion(ndimage.gaussian_filter(sil.astype(float), 6) > 0.5, iterations=int(float(os.environ.get('RING', 0.024)) / STEP))
def in_inner(p):
    i = np.clip(((p[:, 0] - x0) / STEP).astype(int), 0, nx - 1); j = np.clip(((p[:, 2] - z0) / STEP).astype(int), 0, nz - 1); return inner[i, j]
c = face_centres(bk); n = face_normals(bk); mesh_f = in_inner(c)
front = mesh_f & (n[:, 1] < 0); back = mesh_f & (n[:, 1] >= 0)
# smooth the dents: inner vertices onto surfaces fitted to the front and back layers
def basis(x, z):
    return np.c_[np.ones_like(x), x, z, x * x, x * z, z * z, x ** 3, x * x * z, x * z * z, z ** 3]
vn = np.zeros(len(v) * 3); bk.data.vertices.foreach_get('normal', vn); vn = vn.reshape(-1, 3)
iv = in_inner(v)
for layer in (iv & (vn[:, 1] < 0), iv & (vn[:, 1] >= 0)):
    if layer.sum() < 100: continue
    B = basis(v[layer, 0], v[layer, 2]); cf = np.linalg.lstsq(B, v[layer, 1], rcond=None)[0]
    for _ in range(4):
        rr = np.abs(B @ cf - v[layer, 1]); k = rr < np.percentile(rr, 75); cf = np.linalg.lstsq(B[k], v[layer, 1][k], rcond=None)[0]
    v[layer, 1] = B @ cf
# the mesh area has see-through holes in both layers: replace it with two solid sheets (front and back)
# on the fitted surfaces, running a little under the frame ring
fits = []
for layer in (iv & (vn[:, 1] < 0), iv & (vn[:, 1] >= 0)):
    B = basis(v[layer, 0], v[layer, 2]); cf = np.linalg.lstsq(B, v[layer, 1], rcond=None)[0]
    for _ in range(4):
        rr = np.abs(B @ cf - v[layer, 1]); k = rr < np.percentile(rr, 75); cf = np.linalg.lstsq(B[k], v[layer, 1][k], rcond=None)[0]
    fits.append(cf)
bm = bmesh.new(); bm.from_mesh(bk.data); bm.faces.ensure_lookup_table()
bmesh.ops.delete(bm, geom=[f for f, k in zip(bm.faces, mesh_f) if k], context='FACES'); bm.to_mesh(bk.data); bm.free()
sheet = ndimage.binary_dilation(inner, iterations=int(float(os.environ.get('TUCK', 0.008)) / STEP))
ci, cj = np.nonzero(sheet); used = np.zeros((nx + 1, nz + 1), bool)
for di in (0, 1):
    for dj in (0, 1): used[ci + di, cj + dj] = True
ui, uj = np.nonzero(used); corner = np.full((nx + 1, nz + 1), -1); corner[ui, uj] = np.arange(len(ui))
gx, gz = x0 + ui * STEP, z0 + uj * STEP
din = ndimage.distance_transform_edt(inner)[np.clip(ui, 0, nx - 1), np.clip(uj, 0, nz - 1)] * STEP
tuck = float(os.environ.get('TUCK_IN', 0.004)) * np.clip(1 - din / 0.006, 0, 1)        # edges bend inwards, under the ring
quads = np.c_[corner[ci, cj], corner[ci, cj + 1], corner[ci + 1, cj + 1], corner[ci + 1, cj]]
s_ = (gx - gx.min()) / np.ptp(gx); t_ = (gz - gz.min()) / np.ptp(gz)
sheets = []
for k, (cf, mat, box, su, sign) in enumerate(((fits[0], fab_f, box_f, s_, 1), (fits[1], fab_b, box_b, 1 - s_, -1))):
    gy = basis(gx, gz) @ cf + sign * tuck
    me = bpy.data.meshes.new(f'sheet{k}'); me.from_pydata(np.c_[gx, gy, gz].tolist(), [], quads.tolist()); me.update()
    nn = np.zeros(len(quads) * 3); me.polygons.foreach_get('normal', nn)
    if np.sign(nn.reshape(-1, 3)[:, 1].mean()) != -sign: me.flip_normals()       # front sheet faces -y, back sheet faces +y
    me.polygons.foreach_set('use_smooth', np.ones(len(quads), bool))
    u0, u1, v0, v1 = box; lv_ = np.zeros(len(me.loops), int); me.loops.foreach_get('vertex_index', lv_)
    me.uv_layers.new(name='front_uv' if k == 0 else 'back_uv').data.foreach_set('uv', np.c_[u0 + su[lv_] * (u1 - u0), v0 + t_[lv_] * (v1 - v0)].astype(np.float32).reshape(-1))
    me.materials.append(mat); ob = bpy.data.objects.new(f'sheet{k}', me); bpy.context.scene.collection.objects.link(ob); sheets.append(ob)
bpy.ops.object.select_all(action='DESELECT')
for ob in sheets: ob.select_set(True)
bk.select_set(True); bpy.context.view_layer.objects.active = bk; bpy.ops.object.join()
print('backrest mesh faces with real fabric: front', int(front.sum()), 'back', int(back.sum()))

# ---- headrest: its front is a smooth pad; the real headrest photo (mesh teardrops, black centre and rim)
# is wrapped onto the front faces, same scale across and down, anchored at the pad's top edge
HP = os.environ.get('HEAD_PHOTO')
if HP:
    hd = P[9]; hv = verts(hd); c = face_centres(hd); n = face_normals(hd)
    fr = n[:, 1] < -0.3; fv = c[fr]
    hx0, hx1 = np.percentile(fv[:, 0], [0.3, 99.7]); hz1 = np.percentile(fv[:, 2], 99.7)
    img = bpy.data.images.load(HP); ar = img.size[0] / img.size[1]
    lv = np.zeros(len(hd.data.loops), int); hd.data.loops.foreach_get('vertex_index', lv); p = hv[lv]
    uv = np.c_[(p[:, 0] - hx0) / (hx1 - hx0), 1 - (hz1 - p[:, 2]) / (hx1 - hx0) * ar]
    hd.data.uv_layers.new(name='head_uv').data.foreach_set('uv', uv.astype(np.float32).reshape(-1))
    hm, _ = fabric('headrest_photo_fabric', HP, 'head_uv')
    hd.data.materials.append(hm); k = len(hd.data.materials) - 1
    mi = np.zeros(len(c), np.int32); hd.data.polygons.foreach_get('material_index', mi); mi[fr] = k
    hd.data.polygons.foreach_set('material_index', mi); hd.data.update()
    print('headrest front from photo:', int(fr.sum()), 'faces')

# ---- base and column: matte black plastic (Tripo baked chrome reflections into a blotchy texture)
plast = bpy.data.materials.new('frame_plastic'); plast.use_nodes = True; pb = plast.node_tree.nodes['Principled BSDF']
pb.inputs['Base Color'].default_value = (0.03, 0.03, 0.032, 1); pb.inputs['Roughness'].default_value = 0.55; pb.inputs['Specular IOR Level'].default_value = 0.3
for i in (1, 90):
    P[i].data.materials.clear(); P[i].data.materials.append(plast)
if os.environ.get('BASE', 'black') == 'chrome':
    # like the real chair: polished chrome legs, black hub (and black gas-lift column)
    chrome = bpy.data.materials.new('chrome'); chrome.use_nodes = True; cb = chrome.node_tree.nodes['Principled BSDF']
    cb.inputs['Base Color'].default_value = (0.86, 0.86, 0.88, 1); cb.inputs['Metallic'].default_value = 1.0
    cb.inputs['Roughness'].default_value = float(os.environ.get('CHROME_ROUGH', 0.14))
    b = P[1]; b.data.materials.append(chrome)
    c = face_centres(b); legs = np.hypot(c[:, 0] - ax[0], c[:, 1] - ax[1]) > float(os.environ.get('HUB_R', 0.055))
    mi = np.zeros(len(c), np.int32); mi[legs] = 1; b.data.polygons.foreach_set('material_index', mi); b.data.update()
    print('chrome legs:', int(legs.sum()), 'faces; hub stays black')

# ---- regroup into the animation's parts
for g, ids in GROUPS.items():
    obs = [P[i] for i in ids]
    if g == 'wheels':
        for k, o in enumerate(obs): o.name = f'wheel_{k}'
        continue
    bpy.ops.object.select_all(action='DESELECT')
    for o in obs: o.select_set(True)
    bpy.context.view_layer.objects.active = obs[0]
    if len(obs) > 1: bpy.ops.object.join()
    obs[0].name = g
# ---- spine frame: Tripo built it from pieces that meet in a visible step on both arms (about 54 cm up).
# Rebuild it as one seamless surface (voxel remesh), plain black plastic, with the real chair's single
# chrome ring where the Y meets the lower bracket.
fr = bpy.data.objects['frame']
if os.environ.get('SPINE_REMESH', '1') == '1':
    # morphological closing: grow the surface, fuse it into one volume, shrink it back; gaps narrower than
    # twice the grow distance (the cut between the arm pieces) close up, the overall shape stays
    GROW = float(os.environ.get('GROW', 0.003))
    bpy.context.view_layer.objects.active = fr
    for step in ('remesh', 'grow', 'remesh', 'shrink', 'smooth'):          # first remesh: one closed surface with outward normals
        if step in ('grow', 'shrink'):
            md = fr.modifiers.new(step, 'DISPLACE'); md.mid_level = 0.0; md.strength = GROW if step == 'grow' else -GROW; md.direction = 'NORMAL'
        elif step == 'remesh':
            md = fr.modifiers.new(step, 'REMESH'); md.mode = 'VOXEL'; md.voxel_size = float(os.environ.get('VOXEL', 0.0012))
        else:
            md = fr.modifiers.new(step, 'CORRECTIVE_SMOOTH'); md.iterations = 6; md.use_only_smooth = True
        if step == 'shrink':
            fr.data.polygons.foreach_set('use_smooth', np.ones(len(fr.data.polygons), bool)); fr.data.update()
        bpy.ops.object.modifier_apply(modifier=step)
    # fill the groove where the arm pieces met: smooth only the vertices in that band
    z0b, z1b = (float(x) for x in os.environ.get('GROOVE', '0.528,0.562').split(','))
    bm = bmesh.new(); bm.from_mesh(fr.data)
    band = [v for v in bm.verts if z0b < v.co.z < z1b]
    for _ in range(int(os.environ.get('GROOVE_IT', 12))):
        bmesh.ops.smooth_vert(bm, verts=band, factor=0.5, use_axis_x=True, use_axis_y=True, use_axis_z=True)
    bm.to_mesh(fr.data); bm.free(); fr.data.update()
    print('groove band smoothed:', len(band), 'vertices')
    fr.data.materials.clear(); fr.data.materials.append(plast)
    chrome_ring = bpy.data.materials.get('chrome')
    if chrome_ring is None:
        chrome_ring = bpy.data.materials.new('chrome'); chrome_ring.use_nodes = True; cb = chrome_ring.node_tree.nodes['Principled BSDF']
        cb.inputs['Base Color'].default_value = (0.86, 0.86, 0.88, 1); cb.inputs['Metallic'].default_value = 1.0; cb.inputs['Roughness'].default_value = 0.14
    fr.data.materials.append(chrome_ring)
    zr, rh = float(os.environ.get('RING_Z', 0.4555)), float(os.environ.get('RING_H', 0.0035))
    bm = bmesh.new(); bm.from_mesh(fr.data)                 # crisp ring edges: cut the surface along both rims
    for zc in (zr - rh, zr + rh):
        bmesh.ops.bisect_plane(bm, geom=list(bm.verts) + list(bm.edges) + list(bm.faces), plane_co=(0, 0, zc), plane_no=(0, 0, 1))
    bm.to_mesh(fr.data); bm.free(); fr.data.update()
    c = face_centres(fr); ring = np.abs(c[:, 2] - zr) < rh
    mi = np.zeros(len(c), np.int32); mi[ring] = 1; fr.data.polygons.foreach_set('material_index', mi)
    fr.data.polygons.foreach_set('use_smooth', np.ones(len(c), bool)); fr.data.update()
    print('spine rebuilt seamless:', len(c), 'faces; chrome ring faces', int(ring.sum()))
print('objects:', sorted(o.name for o in bpy.data.objects if o.type == 'MESH'))
bpy.ops.file.pack_all(); bpy.ops.wm.save_as_mainfile(filepath=OUT)
print('wrote', OUT)
