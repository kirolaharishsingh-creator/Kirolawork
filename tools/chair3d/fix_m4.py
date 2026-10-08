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

# ---- mechanism: rebuild it as one closed, clean shell (Tripo's surface has pin-holes and split seams),
# then cut out the small bridge Tripo moulded between the two lever paddles and cap both cut faces
me4 = P[4]; bpy.context.view_layer.objects.active = me4
md = me4.modifiers.new('rm', 'REMESH'); md.mode = 'VOXEL'; md.voxel_size = float(os.environ.get('MECH_VOXEL', 0.0008))
bpy.ops.object.modifier_apply(modifier='rm')
md = me4.modifiers.new('sm', 'CORRECTIVE_SMOOTH'); md.iterations = 4; md.use_only_smooth = True
bpy.ops.object.modifier_apply(modifier='sm')
LB = [float(x) for x in os.environ.get('LEVER_BRIDGE', '-0.190,-0.172,-0.002,0.014').split(',')]   # x0,x1,y0,y1
c = face_centres(me4); br = (c[:, 0] > LB[0]) & (c[:, 0] < LB[1]) & (c[:, 1] > LB[2]) & (c[:, 1] < LB[3])
bm = bmesh.new(); bm.from_mesh(me4.data); bm.faces.ensure_lookup_table()
bmesh.ops.delete(bm, geom=[f for f, k in zip(bm.faces, br) if k], context='FACES')
bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
filled = bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary], sides=0)['faces']
bmesh.ops.triangulate(bm, faces=filled)
bm.to_mesh(me4.data); bm.free()
me4.data.polygons.foreach_set('use_smooth', np.ones(len(me4.data.polygons), bool)); me4.data.update()
print('mechanism rebuilt:', len(me4.data.polygons), 'faces; lever bridge removed', int(br.sum()), 'faces, capped with', len(filled))

# ---- materials
def photo_box(img, inset=0.09):
    w, h = img.size; px = np.array(img.pixels[:], np.float32).reshape(h, w, 4)
    ys, xs = np.nonzero(px[..., :3].mean(-1) < 0.6)
    u0, u1, v0, v1 = xs.min() / w, xs.max() / w, ys.min() / h, ys.max() / h
    du, dv = (u1 - u0) * inset, (v1 - v0) * inset
    return u0 + du, u1 - du, v0 + dv, v1 - dv

def fabric(name, path, uvname):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; pb = nt.nodes['Principled BSDF']
    pb.inputs['Roughness'].default_value = 0.9; pb.inputs['Specular IOR Level'].default_value = 0.15   # fabric: no shine
    t = nt.nodes.new('ShaderNodeTexImage'); t.image = bpy.data.images.load(path); t.extension = 'EXTEND'
    u = nt.nodes.new('ShaderNodeUVMap'); u.uv_map = uvname; nt.links.new(u.outputs[0], t.inputs['Vector'])
    # the real photo's fabric as it is: its own colours and stripes, no extra grading (FAB_GAMMA stays 1.0)
    gm = nt.nodes.new('ShaderNodeGamma'); gm.inputs[1].default_value = float(os.environ.get('FAB_GAMMA', 1.0))
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
sheet = ndimage.binary_dilation(inner, iterations=int(float(os.environ.get('TUCK', 0.012)) / STEP))
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
    hd = P[9]
    bm = bmesh.new(); bm.from_mesh(hd.data); bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=2e-4)   # weld Tripo's seam splits first
    bm.to_mesh(hd.data); bm.free(); hd.data.update(); hv = verts(hd)
    # Tripo moulded faint outlines and bumps into the pad (where it pieced it together): settle the front
    # surface onto one smooth fitted curve, blending into the untouched rim
    vn = np.zeros(len(hv) * 3); hd.data.vertices.foreach_get('normal', vn); vn = vn.reshape(-1, 3)
    st = 0.0015; pad = 20; hx0_, hz0_ = hv[:, 0].min() - pad * st, hv[:, 2].min() - pad * st
    gnx, gnz = int(np.ptp(hv[:, 0]) / st) + 2 * pad + 1, int(np.ptp(hv[:, 2]) / st) + 2 * pad + 1
    sil = np.zeros((gnx, gnz), bool); sil[((hv[:, 0] - hx0_) / st).astype(int), ((hv[:, 2] - hz0_) / st).astype(int)] = True
    sil = ndimage.binary_fill_holes(ndimage.binary_closing(sil, iterations=4))
    dist = ndimage.distance_transform_edt(ndimage.gaussian_filter(sil.astype(float), 4) > 0.5) * st
    dv = dist[((hv[:, 0] - hx0_) / st).astype(int), ((hv[:, 2] - hz0_) / st).astype(int)]
    frontv = vn[:, 1] < -0.25
    RIM = float(os.environ.get('HEAD_RIM', 0.010))
    sel = frontv & (dv > RIM)
    B = basis(hv[sel, 0], hv[sel, 2]); cf = np.linalg.lstsq(B, hv[sel, 1], rcond=None)[0]
    for _ in range(4):
        rr = np.abs(B @ cf - hv[sel, 1]); k = rr < np.percentile(rr, 70); cf = np.linalg.lstsq(B[k], hv[sel, 1][k], rcond=None)[0]
    w = np.clip((dv - RIM) / 0.008, 0, 1) * frontv
    hv[:, 1] = hv[:, 1] * (1 - w) + (basis(hv[:, 0], hv[:, 2]) @ cf) * w
    hd.data.vertices.foreach_set('co', hv.reshape(-1)); hd.data.update()
    print('headrest front smoothed:', int((w > 0).sum()), 'vertices')
    # the top edge curls backwards, so the fit above can't reach it: relax its bumps and slots in place
    ztop = hv[:, 2].max()
    bm = bmesh.new(); bm.from_mesh(hd.data); bm.verts.ensure_lookup_table()
    band = [bm.verts[i] for i in np.nonzero((hv[:, 2] > ztop - float(os.environ.get('HEAD_TOP_BAND', 0.035))) & (vn[:, 1] < 0.35))[0]]
    for _ in range(int(os.environ.get('HEAD_TOP_IT', 40))):
        bmesh.ops.smooth_vert(bm, verts=band, factor=0.5, use_axis_x=True, use_axis_y=True, use_axis_z=True)
    bm.to_mesh(hd.data); bm.free(); hd.data.update(); hv = verts(hd)
    print('headrest top edge relaxed:', len(band), 'vertices')
    c = face_centres(hd); n = face_normals(hd)
    fr = n[:, 1] < -float(os.environ.get('HEAD_FACING', 0.5)); fv = c[fr]   # the photo only on the forward-facing pad
    hx0, hx1 = np.percentile(fv[:, 0], [0.3, 99.7]); hz1 = np.percentile(fv[:, 2], 99.7)
    img = bpy.data.images.load(HP); ar = img.size[0] / img.size[1]
    lv = np.zeros(len(hd.data.loops), int); hd.data.loops.foreach_get('vertex_index', lv); p = hv[lv]
    uv = np.c_[(p[:, 0] - hx0) / (hx1 - hx0), np.minimum(1 - (hz1 - p[:, 2]) / (hx1 - hx0) * ar, 0.97)]   # keep off the photo's top rows (wall)
    hd.data.uv_layers.new(name='head_uv').data.foreach_set('uv', uv.astype(np.float32).reshape(-1))
    hm, _ = fabric('headrest_photo_fabric', HP, 'head_uv')
    # the photo is underexposed: its black is far deeper than the soft black fabric of the rim. Lift only
    # the darkest tones to the rim's level (the stripes keep their real look)
    nt = hm.node_tree; pb = nt.nodes['Principled BSDF']; src = pb.inputs['Base Color'].links[0].from_socket
    mx = nt.nodes.new('ShaderNodeMix'); mx.data_type = 'RGBA'; mx.blend_type = 'LIGHTEN'; mx.inputs['Factor'].default_value = 1.0
    lv_ = float(os.environ.get('HEAD_BLACK', 0.02)); mx.inputs[7].default_value = (lv_, lv_, lv_ * 1.05, 1)
    hs = nt.nodes.new('ShaderNodeHueSaturation'); hs.inputs['Saturation'].default_value = 0.0
    nt.links.new(src, hs.inputs['Color']); nt.links.new(hs.outputs['Color'], mx.inputs[6]); nt.links.new(mx.outputs[2], pb.inputs['Base Color'])
    hd.data.materials.append(hm); k = len(hd.data.materials) - 1
    mi = np.zeros(len(c), np.int32); hd.data.polygons.foreach_get('material_index', mi); mi[fr] = k
    hd.data.polygons.foreach_set('material_index', mi); hd.data.update()
    print('headrest front from photo:', int(fr.sum()), 'faces')
    # the real headrest is a thin fabric shield; the spine's plastic loop carries it from behind. Rebuild it
    # cleanly: the outline of its front (seen from the front), the smooth front curve fitted above, and a
    # fabric body ~1.5 cm deep. The plastic-looking back shell is gone. The front and back faces both carry
    # the real headrest photo (from behind the see-through mesh panels show mirrored); the edge is black fabric.
    hv = verts(hd); vn = np.zeros(len(hv) * 3); hd.data.vertices.foreach_get('normal', vn); vn = vn.reshape(-1, 3)
    fv_ = hv[vn[:, 1] < -0.05]
    st = 0.0015; pad = 12; gx0, gz0 = fv_[:, 0].min() - pad * st, fv_[:, 2].min() - pad * st
    gnx, gnz = int(np.ptp(fv_[:, 0]) / st) + 2 * pad + 1, int(np.ptp(fv_[:, 2]) / st) + 2 * pad + 1
    m_ = np.zeros((gnx, gnz), bool); m_[((fv_[:, 0] - gx0) / st).astype(int), ((fv_[:, 2] - gz0) / st).astype(int)] = True
    m_ = ndimage.binary_fill_holes(ndimage.binary_closing(m_, iterations=5))
    m_ = ndimage.binary_erosion(ndimage.gaussian_filter(m_.astype(float), 5) > 0.5, iterations=1)   # smooth outline
    ci, cj = np.nonzero(m_); used = np.zeros((gnx + 1, gnz + 1), bool)
    for di in (0, 1):
        for dj in (0, 1): used[ci + di, cj + dj] = True
    ui, uj = np.nonzero(used); corner = np.full((gnx + 1, gnz + 1), -1); corner[ui, uj] = np.arange(len(ui))
    X, Z = gx0 + ui * st, gz0 + uj * st
    # a gentle curve for the whole shield: a quadratic fitted to the pad's front (no extrapolated curl)
    fr_v = hv[vn[:, 1] < -0.3]
    def b2(x, z): return np.c_[np.ones_like(x), x, z, x * x, x * z, z * z]
    c2 = np.linalg.lstsq(b2(fr_v[:, 0], fr_v[:, 2]), fr_v[:, 1], rcond=None)[0]
    for _ in range(4):
        rr = np.abs(b2(fr_v[:, 0], fr_v[:, 2]) @ c2 - fr_v[:, 1]); kk = rr < np.percentile(rr, 75)
        c2 = np.linalg.lstsq(b2(fr_v[kk, 0], fr_v[kk, 2]), fr_v[kk, 1], rcond=None)[0]
    Y = b2(X, Z) @ c2
    quads = np.c_[corner[ci, cj], corner[ci + 1, cj], corner[ci + 1, cj + 1], corner[ci, cj + 1]]
    sh = bpy.data.meshes.new('headrest_shield'); sh.from_pydata(np.c_[X, Y, Z].tolist(), [], quads.tolist()); sh.update()
    nn = np.zeros(len(quads) * 3); sh.polygons.foreach_get('normal', nn)
    if nn.reshape(-1, 3)[:, 1].mean() > 0: sh.flip_normals()     # front faces look forward (-y)
    # smooth the stair-stepped outline: relax the border ring (and the row inside it) in the x-z plane
    bm = bmesh.new(); bm.from_mesh(sh); bm.verts.ensure_lookup_table()
    edge_v = {v for e in bm.edges if e.is_boundary for v in e.verts}
    ring2 = {w for v in edge_v for e in v.link_edges for w in e.verts} - edge_v
    for it in range(30):
        for vset, f in ((edge_v, 0.5), (ring2, 0.25)):
            new = {}
            for v in vset:
                nb = [e.other_vert(v) for e in v.link_edges if (e.is_boundary or v not in edge_v)]
                if nb:
                    cx = sum(w.co.x for w in nb) / len(nb); cz = sum(w.co.z for w in nb) / len(nb)
                    new[v] = (v.co.x + f * (cx - v.co.x), v.co.z + f * (cz - v.co.z))
            for v, (x, z) in new.items(): v.co.x, v.co.z = x, z
    for v in bm.verts: v.co.y = float(b2(np.array([v.co.x]), np.array([v.co.z])) @ c2)
    bm.to_mesh(sh); bm.free(); sh.update()
    for m in hd.data.materials: sh.materials.append(m)
    sv = np.zeros(len(sh.vertices) * 3); sh.vertices.foreach_get('co', sv); sv = sv.reshape(-1, 3)
    lvs = np.zeros(len(sh.loops), int); sh.loops.foreach_get('vertex_index', lvs)
    uvs = np.c_[(sv[:, 0] - hx0) / (hx1 - hx0), np.minimum(1 - (hz1 - sv[:, 2]) / (hx1 - hx0) * ar, 0.96)][lvs]
    sh.uv_layers.new(name='UVMap'); sh.uv_layers.new(name='head_uv').data.foreach_set('uv', uvs.astype(np.float32).reshape(-1))
    sh.polygons.foreach_set('material_index', np.full(len(quads), k, np.int32))
    old_me = hd.data; hd.data = sh; bpy.data.meshes.remove(old_me)
    nfront = len(quads); bpy.context.view_layer.objects.active = hd
    sd = hd.modifiers.new('shield', 'SOLIDIFY'); sd.thickness = float(os.environ.get('HEAD_DEPTH', 0.015)); sd.offset = -1.0; sd.use_rim = True
    bpy.ops.object.modifier_apply(modifier='shield')
    mi = np.zeros(len(hd.data.polygons), np.int32); hd.data.polygons.foreach_get('material_index', mi)
    mi[2 * nfront:] = 0                                         # the edge: black fabric (slot 0)
    # the back: as on the real chair (seen from behind) one large mesh panel inside a black fabric border.
    # The mesh is a crop of the real headrest's back photo, repeated at its real scale
    HB = os.environ.get('HEAD_BACK_MESH', os.path.join(os.path.dirname(os.path.abspath(HP)), 'headrest_back_mesh_seamless.png'))   # straight stripes built from the real back photo
    if os.path.exists(HB):
        cb_ = face_centres(hd)[nfront:2 * nfront]
        dist = ndimage.distance_transform_edt(m_) * st
        ii = np.clip(((cb_[:, 0] - gx0) / st).astype(int), 0, m_.shape[0] - 1); jj = np.clip(((cb_[:, 2] - gz0) / st).astype(int), 0, m_.shape[1] - 1)
        panel = dist[ii, jj] > float(os.environ.get('HEAD_BORDER', 0.016))
        bmm = bpy.data.materials.new('headrest_back_mesh_fabric'); bmm.use_nodes = True; nt = bmm.node_tree; pb = nt.nodes['Principled BSDF']
        pb.inputs['Roughness'].default_value = 0.9; pb.inputs['Specular IOR Level'].default_value = 0.15
        tx = nt.nodes.new('ShaderNodeTexImage'); tx.image = bpy.data.images.load(HB); tx.extension = 'REPEAT'
        uvn = nt.nodes.new('ShaderNodeUVMap'); uvn.uv_map = 'hb_uv'; nt.links.new(uvn.outputs[0], tx.inputs['Vector'])
        hs = nt.nodes.new('ShaderNodeHueSaturation'); hs.inputs['Saturation'].default_value = 0.0
        nt.links.new(tx.outputs['Color'], hs.inputs['Color']); nt.links.new(hs.outputs['Color'], pb.inputs['Base Color'])
        hd.data.materials.append(bmm); kb = len(hd.data.materials) - 1
        hvv = verts(hd); lvb = np.zeros(len(hd.data.loops), int); hd.data.loops.foreach_get('vertex_index', lvb); pp = hvv[lvb]
        TW, TH = float(os.environ.get('HB_TILE_W', 0.26)), float(os.environ.get('HB_TILE_H', 0.048))   # the crop's real size
        hd.data.uv_layers.new(name='hb_uv').data.foreach_set('uv', np.c_[-pp[:, 0] / TW, pp[:, 2] / TH].astype(np.float32).reshape(-1))
        mi[nfront:2 * nfront] = np.where(panel, kb, 0)
        print('headrest back: mesh panel', int(panel.sum()), 'faces inside a black fabric border')
    hd.data.polygons.foreach_set('material_index', mi)
    hd.data.polygons.foreach_set('use_smooth', np.ones(len(mi), bool)); hd.data.update()
    print('headrest rebuilt as a fabric shield:', nfront, 'front faces + back face + edge; back shell removed')
    HEAD_SHIELD = (m_, gx0, gz0, st, c2, float(os.environ.get('HEAD_DEPTH', 0.015)))   # for keeping the spine loop behind it

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

# ---- clean materials (CLEAN=1): Tripo's colour maps carry painted-in shadows, smudges and seam lines.
# Every remaining part gets a clean material in the real chair's colours; Tripo's normal maps stay, so the
# moulded detail (seams, ribs, grain) is kept. Mesh areas get the real fabric photos.
def clean_from(src, name, colour, rough, spec=0.35, sheen=0.0, weave=0.0):
    m = src.copy(); m.name = name; nt = m.node_tree; pb = [n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'][0]
    # Tripo's normal maps carry dark artefacts (dark blotches on arms, wheels, seat): off unless NORMALS=1
    socks = ['Base Color', 'Metallic', 'Roughness'] + ([] if os.environ.get('NORMALS', '0') == '1' else ['Normal'])
    for sock in socks:
        for l in list(pb.inputs[sock].links): nt.links.remove(l)
    pb.inputs['Base Color'].default_value = (*colour, 1); pb.inputs['Metallic'].default_value = 0.0
    pb.inputs['Roughness'].default_value = rough; pb.inputs['Specular IOR Level'].default_value = spec
    pb.inputs['Sheen Weight'].default_value = sheen; pb.inputs['Sheen Roughness'].default_value = 0.5
    if weave:                                                  # fine woven-fabric grain (bump only)
        tc = nt.nodes.new('ShaderNodeTexCoord'); wv = nt.nodes.new('ShaderNodeTexWave'); wv.inputs['Scale'].default_value = 900.0
        wv.wave_type = 'BANDS'; wv.bands_direction = 'X'; nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 2500.0
        nt.links.new(tc.outputs['Object'], wv.inputs['Vector']); nt.links.new(tc.outputs['Object'], nz.inputs['Vector'])
        ad = nt.nodes.new('ShaderNodeMath'); ad.operation = 'ADD'; nt.links.new(wv.outputs['Fac'], ad.inputs[0]); nt.links.new(nz.outputs['Fac'], ad.inputs[1])
        bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = weave; bp.inputs['Distance'].default_value = 0.0004
        nt.links.new(ad.outputs[0], bp.inputs['Height']); nt.links.new(bp.outputs['Normal'], pb.inputs['Normal'])
    return m

def inner_region(o, ring, sel=None):
    # faces well inside the part's outline seen from the front (x-z), i.e. away from its rim
    v = verts(o); st = 0.0015; pad = 20
    gx0, gz0 = v[:, 0].min() - pad * st, v[:, 2].min() - pad * st
    gnx, gnz = int(np.ptp(v[:, 0]) / st) + 2 * pad + 1, int(np.ptp(v[:, 2]) / st) + 2 * pad + 1
    s = np.zeros((gnx, gnz), bool); s[((v[:, 0] - gx0) / st).astype(int), ((v[:, 2] - gz0) / st).astype(int)] = True
    s = ndimage.binary_fill_holes(ndimage.binary_closing(s, iterations=4))
    s = ndimage.binary_erosion(ndimage.gaussian_filter(s.astype(float), 6) > 0.5, iterations=int(ring / st))
    c = face_centres(o)
    i = np.clip(((c[:, 0] - gx0) / st).astype(int), 0, gnx - 1); j = np.clip(((c[:, 2] - gz0) / st).astype(int), 0, gnz - 1)
    return s[i, j]

def planar_fabric_uvs(o, sel):
    v = verts(o); lv = np.zeros(len(o.data.loops), int); o.data.loops.foreach_get('vertex_index', lv); p = v[lv]
    c = face_centres(o)[sel]; x0_, x1_, z0_, z1_ = c[:, 0].min(), c[:, 0].max(), c[:, 2].min(), c[:, 2].max()
    s = np.clip((p[:, 0] - x0_) / (x1_ - x0_), 0, 1); t = np.clip((p[:, 2] - z0_) / (z1_ - z0_), 0, 1)
    for name, (u0, u1, v0, v1), su in (('front_uv', box_f, s), ('back_uv', box_b, 1 - s)):
        o.data.uv_layers.new(name=name).data.foreach_set('uv', np.c_[u0 + su * (u1 - u0), v0 + t * (v1 - v0)].astype(np.float32).reshape(-1))

def assign(o, sel, mat):
    if mat.name not in [m.name for m in o.data.materials]: o.data.materials.append(mat)
    k = [m.name for m in o.data.materials].index(mat.name)
    mi = np.zeros(len(o.data.polygons), np.int32); o.data.polygons.foreach_get('material_index', mi); mi[sel] = k
    o.data.polygons.foreach_set('material_index', mi); o.data.update()

if os.environ.get('CLEAN', '1') == '1':
    PL = tuple(float(x) for x in os.environ.get('PLASTIC_RGB', '0.026,0.026,0.028').split(','))
    for i in (0, 2, 10, 4, 3, 7, 8):                          # black plastic
        o = P[i]; src = o.data.materials[0]; cm = clean_from(src, f'plastic_{i}', PL, 0.5)
        o.data.materials[0] = cm
    # the headrest is only the fabric-covered shield (its plastic structure is the spine's V arm):
    # rim and edges are soft black fabric, never plastic
    P[9].data.materials[0] = clean_from(P[9].data.materials[0], 'headrest_rim_fabric', (0.02, 0.02, 0.021), 0.92, spec=0.2, sheen=0.3, weave=0.1)
    for i in (7, 8):                                          # armrest pads: soft-touch top
        o = P[i]; c = face_centres(o); pad = c[:, 2] > c[:, 2].max() - 0.028
        assign(o, pad, clean_from(o.data.materials[0], f'pad_{i}', (0.022, 0.022, 0.024), 0.78, spec=0.25, sheen=0.15))
    s5 = P[5]; s5.data.materials[0] = clean_from(s5.data.materials[0], 'seat_fabric', (0.017, 0.017, 0.018), 0.95, spec=0.2, sheen=0.35, weave=0.12)
    for i in GROUPS['wheels']:
        o = P[i]; o.data.materials[0] = clean_from(o.data.materials[0], f'rubber_{i}', (0.02, 0.02, 0.021), 0.55)
    # lumbar mesh, front and back, and the headrest's rear mesh: the same real fabric as the backrest
    for i, ring in ((2, 0.022),):                              # (the headrest's back now mirrors its front)
        o = P[i]; n = face_normals(o); inn = inner_region(o, ring)
        fr_ = inn & (n[:, 1] < -0.2) if i == 2 else np.zeros(len(n), bool)
        bk_ = inn & (n[:, 1] > 0.2)
        if i == 9:                                            # keep the photo-wrapped headrest front as it is
            mi = np.zeros(len(n), np.int32); o.data.polygons.foreach_get('material_index', mi); bk_ &= mi == 0
        planar_fabric_uvs(o, fr_ | bk_)
        if fr_.any(): assign(o, fr_, fab_f)
        assign(o, bk_, fab_b)
        print('fabric on part', i, 'front', int(fr_.sum()), 'back', int(bk_.sum()), 'faces')
    print('clean materials applied')

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
if HP:
    # the spine's loop must stay behind the thin headrest shield (the old thick pad used to hide its top tab)
    m_, gx0, gz0, st, c2, depth = HEAD_SHIELD
    fv = verts(fr); i = ((fv[:, 0] - gx0) / st).astype(int); j = ((fv[:, 2] - gz0) / st).astype(int)
    ok = (i >= 0) & (i < m_.shape[0]) & (j >= 0) & (j < m_.shape[1]); inside = np.zeros(len(fv), bool)
    inside[ok] = ndimage.binary_dilation(m_, iterations=2)[i[ok], j[ok]]
    back = np.c_[np.ones(len(fv)), fv[:, 0], fv[:, 2], fv[:, 0] ** 2, fv[:, 0] * fv[:, 2], fv[:, 2] ** 2] @ c2 + depth + 0.002
    push = inside & (fv[:, 1] < back)
    fv[push, 1] = back[push]; fr.data.vertices.foreach_set('co', fv.reshape(-1)); fr.data.update()
    print('spine loop kept behind the headrest shield:', int(push.sum()), 'vertices moved back')
print('objects:', sorted(o.name for o in bpy.data.objects if o.type == 'MESH'))
bpy.ops.file.pack_all(); bpy.ops.wm.save_as_mainfile(filepath=OUT)
print('wrote', OUT)
