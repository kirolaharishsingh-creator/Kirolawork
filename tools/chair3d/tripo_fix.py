# Clean up the Tripo Studio segmented chair (42 parts) and regroup it into the animation's parts.
# Usage: python3 tripo_fix.py tripo_studio_chair.glb bf.png bb.png out.blend
#   - backrest: the traced V (noisy centre panels and the V strips) is replaced by one smooth panel
#     fitted to the original curve, wrapped with the real fabric photos (front photo in front, back photo behind)
#   - the rear Y spine is cut away from the backrest frame halves so the backrest can lift off the spine
#   - headrest back panels: blotchy texture replaced by the same real fabric
#   - the loose fragment stuck to the seat is deleted; the front leg's caster becomes its own wheel
# Model axes (as exported by Tripo): z up, chair front towards -y, x across.
import bpy, bmesh, sys, os, numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

GLB, FRONT, BACK, OUT = sys.argv[-4:]
GROUPS = {
    'headrest': [32, 22, 7, 41, 30, 39, 40, 10],
    'backrest': [5, 23],                          # frame halves; the spine is cut off them below
    'frame': [11, 25, 29, 28],
    'lumbar': [16, 27, 8, 6, 21],
    'seat': [0],
    'arm_l': [1], 'arm_r': [4],
    'mechanism': [26, 36, 37],
    'gas_lift': [9],
    'base': [24, 12, 14, 15, 13],
    'wheels': [18, 20, 31, 19],
}
PANEL = [2, 3, 33, 34, 35, 17, 38]               # backrest mesh area with the traced V: replaced
HEAD_BACK = [39, 40]                              # headrest rear mesh panels: re-skinned
PLAIN = [5, 23, 7, 30, 41]                       # frames whose inner lip carries baked mesh stripes: plain black plastic
LUMBAR_MESH = [16, 27]                            # lumbar mesh panels: traced dents smoothed out, real fabric
LUMBAR_FRAMES = [8, 6]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=GLB)
P = {int(o.name.split('_')[-1].split('.')[0]): o for o in bpy.data.objects if o.type == 'MESH'}
for o in P.values():                              # bake transforms so all maths is in world space
    o.data.transform(o.matrix_world); o.matrix_world.identity(); o.parent = None

def verts(o):
    v = np.zeros(len(o.data.vertices) * 3); o.data.vertices.foreach_get('co', v); return v.reshape(-1, 3)

def face_centres(o):
    c = np.zeros(len(o.data.polygons) * 3); o.data.polygons.foreach_get('center', c); return c.reshape(-1, 3)

def components(o):
    m = o.data; bm = bmesh.new(); bm.from_mesh(m); bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=2e-4); bm.to_mesh(m); bm.free()
    nf = len(m.polygons); lt = np.zeros(nf, int); m.polygons.foreach_get('loop_total', lt)
    lv = np.zeros(len(m.loops), int); m.loops.foreach_get('vertex_index', lv); nv = len(m.vertices)
    G = coo_matrix((np.ones(len(lv)), (np.repeat(np.arange(nf), lt), nf + lv)), shape=(nf + nv, nf + nv))
    return connected_components(G, directed=False)[1][:nf]

def split_faces(o, sel, name):
    # move the selected faces of o into a new object called name (materials and UVs come along)
    new = o.copy(); new.data = o.data.copy(); new.name = name; bpy.context.scene.collection.objects.link(new)
    for ob, keep in ((o, ~sel), (new, sel)):
        bm = bmesh.new(); bm.from_mesh(ob.data); bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[f for f, k in zip(bm.faces, keep) if not k], context='FACES')
        bm.to_mesh(ob.data); bm.free()
    return new

# ---- loose fragment on the seat
fc = components(P[0]); big = np.bincount(fc).argmax()
bm = bmesh.new(); bm.from_mesh(P[0].data); bm.faces.ensure_lookup_table()
bmesh.ops.delete(bm, geom=[f for f, c in zip(bm.faces, fc) if c != big], context='FACES'); bm.to_mesh(P[0].data); bm.free()
print('seat fragment removed:', int((fc != big).sum()), 'faces')

# ---- front leg (13): its caster becomes a separate wheel
c13 = face_centres(P[13]); wsel = (c13[:, 2] < 0.056) & (c13[:, 1] < c13[:, 1].min() + 0.07)
wheel5 = split_faces(P[13], wsel, 'tripo_part_99'); P[99] = wheel5; GROUPS['wheels'].append(99)
print('front caster split off:', int(wsel.sum()), 'faces')

# ---- backrest panel: one smooth surface fitted to the original mesh area
pv = np.vstack([verts(P[i]) for i in PANEL])
def basis(x, z):
    return np.c_[np.ones_like(x), x, z, x * x, x * z, z * z, x ** 3, x * x * z, x * z * z, z ** 3, x ** 4, z ** 4]
coef = np.linalg.lstsq(basis(pv[:, 0], pv[:, 2]), pv[:, 1], rcond=None)[0]
for _ in range(4):                                # refit without the V ridges (largest residuals)
    r = np.abs(basis(pv[:, 0], pv[:, 2]) @ coef - pv[:, 1]); k = r < np.percentile(r, 75)
    coef = np.linalg.lstsq(basis(pv[k, 0], pv[k, 2]), pv[k, 1], rcond=None)[0]
surf = lambda x, z: basis(np.atleast_1d(x), np.atleast_1d(z)) @ coef
STEP = float(os.environ.get('GRID', 0.0015))
PAD = int(0.03 / STEP)
x0, z0 = pv[:, 0].min() - PAD * STEP, pv[:, 2].min() - PAD * STEP
nx, nz = int(np.ptp(pv[:, 0]) / STEP) + 2 * PAD + 1, int(np.ptp(pv[:, 2]) / STEP) + 2 * PAD + 1
occ = np.zeros((nx, nz), bool); occ[((pv[:, 0] - x0) / STEP).astype(int), ((pv[:, 2] - z0) / STEP).astype(int)] = True
mask = ndimage.binary_fill_holes(ndimage.binary_closing(occ, iterations=int(0.009 / STEP)))
mask = ndimage.binary_opening(mask, iterations=int(0.006 / STEP))            # drop thin stray bits along the edge
# the opening inside the frame, with a smooth outline (Tripo's own edge is ragged)
opening = ndimage.binary_erosion(ndimage.gaussian_filter(mask.astype(float), 0.012 / STEP) > 0.5, iterations=max(1, int(0.006 / STEP)))
pmask = ndimage.binary_dilation(opening, iterations=int(0.015 / STEP))   # the panel runs on under the frame lip
ci, cj = np.nonzero(pmask)
corner = np.full((nx + 1, nz + 1), -1); used = np.zeros((nx + 1, nz + 1), bool)
for di in (0, 1):
    for dj in (0, 1): used[ci + di, cj + dj] = True
ui, uj = np.nonzero(used); corner[ui, uj] = np.arange(len(ui))
gx, gz = x0 + ui * STEP, z0 + uj * STEP; gy = surf(gx, gz)
# the band that runs on under the frame bends backwards, so it always stays behind the frame lip
din = ndimage.distance_transform_edt(opening) - ndimage.distance_transform_edt(~opening)   # cells inside (+) / outside (-) the opening
dv = din[np.clip(ui, 0, nx - 1), np.clip(uj, 0, nz - 1)]
gy = gy + float(os.environ.get('TUCK_BACK', 0.006)) * np.clip((0.006 - dv * STEP) / 0.012, 0, 1)
quads = np.c_[corner[ci, cj], corner[ci, cj + 1], corner[ci + 1, cj + 1], corner[ci + 1, cj]]   # wound so normals face -y (front)
pm = bpy.data.meshes.new('backrest_panel'); pm.from_pydata(np.c_[gx, gy, gz].tolist(), [], quads.tolist()); pm.update()
pm.polygons.foreach_set('use_smooth', np.ones(len(quads), bool))
nrm = np.zeros(len(quads) * 3); pm.polygons.foreach_get('normal', nrm)
if nrm.reshape(-1, 3)[:, 1].mean() > 0: pm.flip_normals()
panel = bpy.data.objects.new('backrest_panel', pm); bpy.context.scene.collection.objects.link(panel)
print('backrest panel', len(quads), 'quads, fit rms', round(float(np.sqrt(np.mean((surf(pv[:, 0], pv[:, 2]) - pv[:, 1]) ** 2))), 4))

# ---- rear spine: Tripo built the Y spine into the backrest frame halves. Faces inside the panel opening
# (away from the outer ring) and faces well behind the panel are the spine: they move to the frame group,
# and any spine surface in front of the mesh is pushed just behind it (in the real chair the spine is behind the mesh)
GAP = float(os.environ.get('SPINE_GAP', 0.03)); BEHIND = 0.012
inner = ndimage.binary_erosion(mask, iterations=int(0.015 / STEP))
def in_grid(p, g):
    i = ((p[:, 0] - x0) / STEP).astype(int); j = ((p[:, 2] - z0) / STEP).astype(int)
    ok = (i >= 0) & (i < nx) & (j >= 0) & (j < nz); r = np.zeros(len(p), bool); r[ok] = g[i[ok], j[ok]]; return r
in_opening = lambda p: in_grid(p, inner)
for i in (5, 23):
    c = face_centres(P[i]); spine = in_opening(c) | (c[:, 1] > surf(c[:, 0], c[:, 2]) + GAP)
    if not spine.any(): continue
    s = split_faces(P[i], spine, f'tripo_part_{100 + i}'); P[100 + i] = s; GROUPS['frame'].append(100 + i)
    v = verts(s); lim = surf(v[:, 0], v[:, 2]) + BEHIND; push = in_opening(v) & (v[:, 1] < lim)
    v[push, 1] = lim[push]; s.data.vertices.foreach_set('co', v.reshape(-1)); s.data.update()
    print('spine cut from part', i, int(spine.sum()), 'faces;', int(push.sum()), 'vertices moved behind the mesh')
    # what is left of the frame half must stop at the smooth opening: ragged bits reaching into it go
    c = face_centres(P[i]); cut = in_grid(c, opening)
    bm = bmesh.new(); bm.from_mesh(P[i].data); bm.faces.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[f for f, k in zip(bm.faces, cut) if k], context='FACES'); bm.to_mesh(P[i].data); bm.free()
    print('  frame trimmed to the opening:', int(cut.sum()), 'faces')

# ---- real fabric material (front photo seen from the front, back photo from behind)
def photo_box(img, inset=0.09):
    w, h = img.size; px = np.array(img.pixels[:], np.float32).reshape(h, w, 4)
    ys, xs = np.nonzero(px[..., :3].mean(-1) < 0.6)
    u0, u1, v0, v1 = xs.min() / w, xs.max() / w, ys.min() / h, ys.max() / h
    du, dv = (u1 - u0) * inset, (v1 - v0) * inset
    return u0 + du, u1 - du, v0 + dv, v1 - dv
imf, imb = bpy.data.images.load(FRONT), bpy.data.images.load(BACK)
bf, bb = photo_box(imf), photo_box(imb)

def fabric_material(name):
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree; pb = nt.nodes['Principled BSDF']
    pb.inputs['Roughness'].default_value = 0.7; pb.inputs['Specular IOR Level'].default_value = 0.25
    tf, tb = nt.nodes.new('ShaderNodeTexImage'), nt.nodes.new('ShaderNodeTexImage')
    for t, im, uvn in ((tf, imf, 'front_uv'), (tb, imb, 'back_uv')):
        t.image = im; t.extension = 'EXTEND'; u = nt.nodes.new('ShaderNodeUVMap'); u.uv_map = uvn; nt.links.new(u.outputs[0], t.inputs['Vector'])
    geo = nt.nodes.new('ShaderNodeNewGeometry'); mx = nt.nodes.new('ShaderNodeMix'); mx.data_type = 'RGBA'
    nt.links.new(geo.outputs['Backfacing'], mx.inputs['Factor']); nt.links.new(tf.outputs['Color'], mx.inputs[6]); nt.links.new(tb.outputs['Color'], mx.inputs[7])
    gm = nt.nodes.new('ShaderNodeGamma'); gm.inputs[1].default_value = float(os.environ.get('FAB_GAMMA', 1.25)); nt.links.new(mx.outputs[2], gm.inputs[0])
    nt.links.new(gm.outputs[0], pb.inputs['Base Color']); return m

def planar_uvs(o, xr, zr):
    m = o.data; lv = np.zeros(len(m.loops), int); m.loops.foreach_get('vertex_index', lv); p = verts(o)[lv]
    s = np.clip((p[:, 0] - xr[0]) / (xr[1] - xr[0]), 0, 1); t = np.clip((p[:, 2] - zr[0]) / (zr[1] - zr[0]), 0, 1)
    for name, (u0, u1, v0, v1), su in (('front_uv', bf, s), ('back_uv', bb, 1 - s)):   # from behind, left and right swap
        lay = m.uv_layers.new(name=name); lay.data.foreach_set('uv', np.c_[u0 + su * (u1 - u0), v0 + t * (v1 - v0)].astype(np.float32).reshape(-1))

fab = fabric_material('backrest_fabric')
planar_uvs(panel, (gx.min(), gx.max()), (gz.min(), gz.max())); pm.materials.append(fab)
hb = np.vstack([verts(P[i]) for i in HEAD_BACK]); hfab = fabric_material('headrest_back_fabric')
for i in HEAD_BACK:
    planar_uvs(P[i], (hb[:, 0].min(), hb[:, 0].max()), (hb[:, 2].min(), hb[:, 2].max()))
    P[i].data.materials.clear(); P[i].data.materials.append(hfab)

def flatten(o):
    # smooth the traced V dents out of a mesh panel: every vertex goes onto a surface fitted to the panel
    v = verts(o); B = basis(v[:, 0], v[:, 2]); c = np.linalg.lstsq(B, v[:, 1], rcond=None)[0]
    for _ in range(4):
        r = np.abs(B @ c - v[:, 1]); k = r < np.percentile(r, 70); c = np.linalg.lstsq(B[k], v[k, 1], rcond=None)[0]
    v[:, 1] = B @ c; o.data.vertices.foreach_set('co', v.reshape(-1)); o.data.update()

lb = np.vstack([verts(P[i]) for i in LUMBAR_MESH])
for i in LUMBAR_MESH:
    flatten(P[i])
    planar_uvs(P[i], (lb[:, 0].min(), lb[:, 0].max()), (lb[:, 2].min(), lb[:, 2].max()))
    P[i].data.materials.clear(); P[i].data.materials.append(fab)
# the lumbar frames' inner lip carries baked mesh stripes around each window: the front faces inside the
# windows' outline get the same real fabric, so each window reads as one even mesh panel
LS = 0.002; lx0, lz0 = lb[:, 0].min() - 0.03, lb[:, 2].min() - 0.03
win = np.zeros((int((np.ptp(lb[:, 0]) + 0.06) / LS) + 1, int((np.ptp(lb[:, 2]) + 0.06) / LS) + 1), bool)
win[((lb[:, 0] - lx0) / LS).astype(int), ((lb[:, 2] - lz0) / LS).astype(int)] = True
win = ndimage.binary_dilation(ndimage.binary_fill_holes(ndimage.binary_closing(win, iterations=4)), iterations=int(os.environ.get('LIP_CELLS', 3)))
for i in LUMBAR_FRAMES:
    o = P[i]; c = face_centres(o); n = np.zeros(len(c) * 3); o.data.polygons.foreach_get('normal', n); n = n.reshape(-1, 3)
    gi = np.clip(((c[:, 0] - lx0) / LS).astype(int), 0, win.shape[0] - 1); gj = np.clip(((c[:, 2] - lz0) / LS).astype(int), 0, win.shape[1] - 1)
    lip = win[gi, gj] & (n[:, 1] < -0.3)
    planar_uvs(o, (lb[:, 0].min(), lb[:, 0].max()), (lb[:, 2].min(), lb[:, 2].max()))
    o.data.materials.append(fab); k = len(o.data.materials) - 1
    mi = np.zeros(len(c), np.int32); o.data.polygons.foreach_get('material_index', mi); mi[lip] = k
    o.data.polygons.foreach_set('material_index', mi); o.data.update()
    print('lumbar frame', i, 'lip faces with fabric:', int(lip.sum()))

# plain black plastic for the frames, at the median colour of their own texture so they still match the rest
plast = None
for i in PLAIN:
    m0 = P[i].data.materials[0]; tex = [n for n in m0.node_tree.nodes if n.type == 'TEX_IMAGE'][0].image
    px = np.array(tex.pixels[:], np.float32).reshape(-1, 4)[::97, :3]; dark = px[px.mean(1) < np.percentile(px.mean(1), 60)]
    col = np.median(dark, 0)
    if plast is None:
        plast = bpy.data.materials.new('frame_plastic'); plast.use_nodes = True; pb = plast.node_tree.nodes['Principled BSDF']
        pb.inputs['Base Color'].default_value = (*col, 1); pb.inputs['Roughness'].default_value = 0.6; pb.inputs['Specular IOR Level'].default_value = 0.3
        print('frame plastic colour', col.round(3))
    P[i].data.materials.clear(); P[i].data.materials.append(plast)

# ---- regroup: delete the replaced panels, join the rest into one object per animation part
for i in PANEL: bpy.data.objects.remove(P.pop(i))
GROUPS['backrest_panel'] = []
for g, ids in GROUPS.items():
    obs = [P[i] for i in ids if i in P]
    if g == 'backrest_panel': obs = [panel]
    if g == 'wheels':
        for k, o in enumerate(obs): o.name = f'wheel_{k}'
        continue
    bpy.ops.object.select_all(action='DESELECT')
    for o in obs: o.select_set(True)
    bpy.context.view_layer.objects.active = obs[0]
    if len(obs) > 1: bpy.ops.object.join()
    obs[0].name = g
bpy.ops.object.select_all(action='DESELECT')
bk, bp = bpy.data.objects['backrest'], bpy.data.objects['backrest_panel']
bk.select_set(True); bp.select_set(True); bpy.context.view_layer.objects.active = bk; bpy.ops.object.join()
# loose bits: small pieces of the backrest go to the spine frame, tiny splinters of the frame go
fr = bpy.data.objects['frame']
fc = components(bk); n = np.bincount(fc); cz = face_centres(bk)[:, 2]
low = np.array([cz[fc == k].mean() < gz.min() for k in range(len(n))])     # pieces hanging below the backrest belong to the spine
small = (n[fc] < 0.02 * len(fc)) | low[fc]
if small.any():
    bit = split_faces(bk, small, 'bits'); bpy.ops.object.select_all(action='DESELECT')
    fr.select_set(True); bit.select_set(True); bpy.context.view_layer.objects.active = fr; bpy.ops.object.join()
    print('backrest bits moved to frame:', int(small.sum()), 'faces')
fc = components(fr); n = np.bincount(fc); tiny = n[fc] < int(os.environ.get('SPLINTER', 1500))
bm = bmesh.new(); bm.from_mesh(fr.data); bm.faces.ensure_lookup_table()
bmesh.ops.delete(bm, geom=[f for f, k in zip(bm.faces, tiny) if k], context='FACES'); bm.to_mesh(fr.data); bm.free()
print('frame splinters removed:', int(tiny.sum()), 'faces in', len(np.unique(fc[tiny])), 'pieces')
print('objects:', sorted(o.name for o in bpy.data.objects if o.type == 'MESH'))
bpy.ops.file.pack_all(); bpy.ops.wm.save_as_mainfile(filepath=OUT)   # .blend keeps the photo materials (glTF would drop them)
print('wrote', OUT)
