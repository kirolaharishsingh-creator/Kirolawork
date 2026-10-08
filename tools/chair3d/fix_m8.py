# Prepare Tripo model 8 ("final last one.glb", 19 parts, textured) for the exploded animation (anim3.py).
# Usage: python3 fix_m8.py model.glb mesh_tile.png out.blend
#   - mesh panels (backrest, lumbar, headrest): Tripo modelled the mesh as coarse, irregular tunnels through a
#     thick sheet (background and spine show through as white speckles). They are closed into smooth panels
#     (voxel remesh + morphological closing), spliced into the original part along a smooth outline, and
#     covered with the real chair's mesh fabric (a seamless tile from the real photo)
#   - backrest part 0 also carries the spine: split off by depth (the spine runs behind the panel)
#   - clean materials in the real chair's true black; chrome base legs with a black hub; black rubber wheels
#   - parts renamed to anim3.py's names
# Model axes (as exported by Tripo): z up, chair front towards -y, x across.
import bpy, bmesh, sys, os, mathutils, numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

GLB, TILE, OUT = sys.argv[-3:]
GROUPS = {'headrest': [2], 'backrest': [0], 'frame': [], 'lumbar': [1], 'seat': [3], 'arm_l': [5], 'arm_r': [6],
          'mechanism': [12, 14, 7], 'gas_lift': [13], 'base': [4, 17], 'wheels': [9, 8, 10, 15, 11]}

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=GLB)
P = {}
for o in [o for o in bpy.data.objects if o.type == 'MESH']:
    o.data.transform(o.matrix_world); o.matrix_world.identity(); o.parent = None
    i = int(o.name.split('_')[-1].split('.')[0])
    if len(o.data.vertices) < 10: bpy.data.objects.remove(o); continue   # part 18: 3 stray vertices
    P[i] = o
# part 16 is the other half of caster 15 (same spot): one wheel
bpy.ops.object.select_all(action='DESELECT'); P[16].select_set(True); P[15].select_set(True)
bpy.context.view_layer.objects.active = P[15]; bpy.ops.object.join(); del P[16]

def verts(o):
    v = np.zeros(len(o.data.vertices) * 3); o.data.vertices.foreach_get('co', v); return v.reshape(-1, 3)

def face_centres(o):
    c = np.zeros(len(o.data.polygons) * 3); o.data.polygons.foreach_get('center', c); return c.reshape(-1, 3)

def apply(o, md):
    bpy.context.view_layer.objects.active = o; bpy.ops.object.modifier_apply(modifier=md.name)

def delete_faces(o, sel):
    bm = bmesh.new(); bm.from_mesh(o.data); bm.faces.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[f for f, k in zip(bm.faces, sel) if k], context='FACES'); bm.to_mesh(o.data); bm.free(); o.data.update()

def mask2d(p, st, pad=30):
    g = (p[:, 0].min() - pad * st, p[:, 2].min() - pad * st)
    n = (int(np.ptp(p[:, 0]) / st) + 2 * pad + 1, int(np.ptp(p[:, 2]) / st) + 2 * pad + 1)
    def ij(q): return np.clip(((q[:, 0] - g[0]) / st).astype(int), 0, n[0] - 1), np.clip(((q[:, 2] - g[1]) / st).astype(int), 0, n[1] - 1)
    return n, ij

# ---- mesh panels: close the tunnels, splice the closed panel into the original part
GROW, VOX, D = float(os.environ.get('GROW', 0.0025)), float(os.environ.get('VOXEL', 0.0012)), float(os.environ.get('D', 0.0035))
E, OV, ST = float(os.environ.get('E', 0.002)), float(os.environ.get('OV', 0.002)), 0.001
PANEL = {}
for i in (0, 1, 2):
    o = P[i]; cl = o.copy(); cl.data = o.data.copy(); bpy.context.scene.collection.objects.link(cl)
    for step in ('remesh', 'grow', 'remesh', 'shrink', 'smooth'):
        if step in ('grow', 'shrink'):
            md = cl.modifiers.new(step, 'DISPLACE'); md.mid_level = 0.0; md.strength = GROW if step == 'grow' else -GROW; md.direction = 'NORMAL'
        elif step == 'remesh':
            md = cl.modifiers.new(step, 'REMESH'); md.mode = 'VOXEL'; md.voxel_size = VOX
        else:
            md = cl.modifiers.new(step, 'CORRECTIVE_SMOOTH'); md.iterations = 6; md.use_only_smooth = True
        apply(cl, md)
    # the closing keeps the wide panels but eats thin rims and struts: keep only its big pieces
    bm = bmesh.new(); bm.from_mesh(cl.data); bm.faces.ensure_lookup_table(); seen = set(); comps = []
    for f in bm.faces:
        if f.index in seen: continue
        stack, comp = [f], []; seen.add(f.index)
        while stack:
            x = stack.pop(); comp.append(x)
            for e in x.edges:
                for y in e.link_faces:
                    if y.index not in seen: seen.add(y.index); stack.append(y)
        comps.append(comp)
    bmesh.ops.delete(bm, geom=[f for c in comps if len(c) < 0.05 * len(bm.faces) for f in c], context='FACES')
    bm.to_mesh(cl.data); bm.free()
    vo, vc = verts(o), verts(cl)
    # smooth front (x-z) outline of the panel: the seam between panel and rim follows this curve
    n, ij = mask2d(vc, ST); m = np.zeros(n, bool); m[ij(vc)] = True
    m = ndimage.gaussian_filter(ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=3)).astype(float), 8) > 0.5
    ero = lambda k: ndimage.binary_erosion(m, iterations=k) if k > 0 else m      # (scipy: 0 iterations erodes to nothing)
    inner, outer = ero(int(E / ST)), ero(max(int(E / ST) - int(OV / ST), 0))
    dvo, _ = cKDTree(vc).query(vo); near = dvo < D
    oc = face_centres(o); fv = [np.array(p.vertices) for p in o.data.polygons]
    gone = np.array([near[f].all() for f in fv]) & inner[ij(oc)]
    keep = outer[ij(face_centres(cl))]
    delete_faces(o, gone); delete_faces(cl, ~keep)
    cl.data.polygons.foreach_set('use_smooth', np.ones(len(cl.data.polygons), bool))
    PANEL[i] = (n, ij, m)
    cl.data.materials.clear(); [cl.data.materials.append(x) for x in o.data.materials]
    pm = bpy.data.materials.new(f'mesh_panel_{i}'); o.data.materials.append(pm); cl.data.materials.append(pm)
    k = len(o.data.materials) - 1; cl.data.polygons.foreach_set('material_index', np.full(len(cl.data.polygons), k, np.int32))
    bpy.ops.object.select_all(action='DESELECT'); cl.select_set(True); o.select_set(True); bpy.context.view_layer.objects.active = o
    bpy.ops.object.join()
    print('panel', i, ': removed', int(gone.sum()), 'tunnel faces, spliced', int(keep.sum()), 'closed faces')

# ---- spine: part 0 is backrest + spine in one piece. The spine runs behind the backrest (front is -y):
# faces deeper than the backrest's own depth behind its panel, or outside its outline, are the spine
o = P[0]; n, ij, m = PANEL[0]
bm = bmesh.new(); bm.from_mesh(o.data); bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5); bm.to_mesh(o.data); bm.free(); o.data.update()
mi = np.zeros(len(o.data.polygons), np.int32); o.data.polygons.foreach_get('material_index', mi)
c = face_centres(o); pan = mi == len(o.data.materials) - 1
ys = np.zeros(n); cnt = np.zeros(n); a, b = ij(c[pan]); np.add.at(ys, (a, b), c[pan, 1]); np.add.at(cnt, (a, b), 1)
have = cnt > 0; ymap = np.where(have, ys / np.maximum(cnt, 1), 0)
idx = ndimage.distance_transform_edt(~have, return_distances=False, return_indices=True); ymap = ymap[idx[0], idx[1]]
RIM, DEPTH = float(os.environ.get('RIM', 0.03)), float(os.environ.get('DEPTH', 0.022))
foot = ndimage.binary_dilation(m, iterations=int(RIM / ST))
a, b = ij(c); spine = ~foot[a, b] | (c[:, 1] > ymap[a, b] + DEPTH)
fr = o.copy(); fr.data = o.data.copy(); fr.name = 'spine'; bpy.context.scene.collection.objects.link(fr)
delete_faces(o, spine); delete_faces(fr, ~spine)
for ob in (o, fr):            # cap the cut openings
    bm = bmesh.new(); bm.from_mesh(ob.data); bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
    bm.to_mesh(ob.data); bm.free(); ob.data.update()
# small loose islands left by the cut (bracket stubs) join whichever piece they sit closest to
def islands(ob):
    bm = bmesh.new(); bm.from_mesh(ob.data); bm.faces.ensure_lookup_table(); seen = np.zeros(len(bm.faces), bool); out = []
    for f in bm.faces:
        if seen[f.index]: continue
        stack, comp = [f], [f.index]; seen[f.index] = True
        while stack:
            x = stack.pop()
            for e in x.edges:
                for y in e.link_faces:
                    if not seen[y.index]: seen[y.index] = True; stack.append(y); comp.append(y.index)
        out.append(np.array(comp))
    bm.free(); return out
moved = 0
for src, dst in ((o, fr),):                 # (the backrest is one piece; the spine keeps its brackets)
    isl = islands(src); big = max(len(x) for x in isl); c = face_centres(src); cd = face_centres(dst)
    small = [x for x in isl if len(x) < 0.05 * big]
    if not small: continue
    main = np.ones(len(c), bool)
    for x in small: main[x] = False
    t_own, t_dst = cKDTree(c[main]), cKDTree(cd); go = np.zeros(len(c), bool)
    for x in small:
        if t_dst.query(c[x])[0].min() < t_own.query(c[x])[0].min(): go[x] = True
    if go.any():
        piece = src.copy(); piece.data = src.data.copy(); bpy.context.scene.collection.objects.link(piece)
        delete_faces(src, go); delete_faces(piece, ~go); moved += int(go.sum())
        bpy.ops.object.select_all(action='DESELECT'); piece.select_set(True); dst.select_set(True); bpy.context.view_layer.objects.active = dst; bpy.ops.object.join()
# pieces left loose on the spine (bracket ends cut by the split) would float on their own: each joins the part
# it touches (backrest or lumbar), otherwise it is dropped
isl = islands(fr); big = max(len(x) for x in isl); c = face_centres(fr)
targets = [(o, cKDTree(face_centres(o))), (P[1], cKDTree(face_centres(P[1])))]
give = {id(o): np.zeros(len(c), bool), id(P[1]): np.zeros(len(c), bool)}; stub = np.zeros(len(c), bool)
for x in isl:
    if len(x) >= 0.05 * big: continue
    d = [t.query(c[x])[0].min() for _, t in targets]; k = int(np.argmin(d))
    if d[k] < 0.01: give[id(targets[k][0])][x] = True
    else: stub[x] = True
for dst, _ in targets:
    sel = give[id(dst)]
    if not sel.any(): continue
    piece = fr.copy(); piece.data = fr.data.copy(); bpy.context.scene.collection.objects.link(piece); delete_faces(piece, ~sel)
    piece.data.materials.clear(); [piece.data.materials.append(x) for x in dst.data.materials]
    piece.data.polygons.foreach_set('material_index', np.zeros(len(piece.data.polygons), np.int32))
    bpy.ops.object.select_all(action='DESELECT'); piece.select_set(True); dst.select_set(True); bpy.context.view_layer.objects.active = dst; bpy.ops.object.join()
moved_out = sum(int(v.sum()) for v in give.values())
delete_faces(fr, stub | give[id(o)] | give[id(P[1])])
print('split tidy: moved', moved, 'faces of loose islands; spine pieces given to backrest/lumbar', moved_out, 'dropped', int(stub.sum()))
P[80] = fr; GROUPS['frame'].append(80)
print('spine split:', int(spine.sum()), 'faces')

# ---- crumbs: tiny loose islands left at the panel seams would float as specks once the parts fly apart
for i in (0, 1, 2):
    o = P[i]; bm = bmesh.new(); bm.from_mesh(o.data); bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5); bm.to_mesh(o.data); bm.free()
    isl = islands(o); big = max(len(x) for x in isl); crumbs = np.zeros(len(o.data.polygons), bool)
    for x in isl:
        if len(x) < float(os.environ.get('CRUMB', 0.005)) * big: crumbs[x] = True
    delete_faces(o, crumbs); print('part', i, 'crumbs removed:', int(crumbs.sum()), 'faces')

# ---- thin rods: Tripo hung thin rods off the backrest's bottom corners and the lumbar's top corners (not on the
# real chair). Seen from the front they are a few mm wide: a morphological opening of the part's front outline
# removes them, and faces outside the opened outline in that band go
def drop_rods(o, zlo, zhi, r=float(os.environ.get('ROD_R', 0.0025))):
    v = verts(o); n, ij = mask2d(v, ST); m = np.zeros(n, bool); m[ij(v)] = True
    m = ndimage.binary_closing(m, iterations=2)
    op = ndimage.binary_opening(m, structure=np.ones((3, 3)), iterations=int(r / ST))
    c = face_centres(o); band = (c[:, 2] > zlo) & (c[:, 2] < zhi); rod = band & ~op[ij(c)]
    delete_faces(o, rod)
    bm = bmesh.new(); bm.from_mesh(o.data); bmesh.ops.delete(bm, geom=[x for x in bm.verts if not x.link_faces], context='VERTS'); bm.to_mesh(o.data); bm.free()
    return int(rod.sum())
print('rods removed: backrest', drop_rods(P[0], 0.0, 0.62), 'faces, lumbar', drop_rods(P[1], 0.55, 1.0), 'faces')

# the backrest's bottom-right corner also carries two short rods (1 cm) mixed in with the rim: cut that corner
# below the rim line and cap it
RC = [float(x) for x in os.environ.get('ROD_CORNER', '0.110,0.137,0.5915').split(',')]       # x0, x1, z below which to cut
o = P[0]; bm = bmesh.new(); bm.from_mesh(o.data)
bmesh.ops.bisect_plane(bm, geom=list(bm.verts) + list(bm.edges) + list(bm.faces), plane_co=(0, 0, RC[2]), plane_no=(0, 0, 1))
bm.faces.ensure_lookup_table()
cut = [f for f in bm.faces if RC[0] < f.calc_center_median().x < RC[1] and f.calc_center_median().z < RC[2]]
bmesh.ops.delete(bm, geom=cut, context='FACES'); bmesh.ops.delete(bm, geom=[x for x in bm.verts if not x.link_faces], context='VERTS')
bm.to_mesh(o.data); bm.free(); o.data.update(); print('rod corner cut:', len(cut), 'faces')

# ---- pin-holes: small open holes in the rims let the backdrop through as white specks; cap them
def fill_small_holes(o, maxlen=float(os.environ.get('HOLE_MAX', 0.03))):
    bm = bmesh.new(); bm.from_mesh(o.data); bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5); bm.edges.ensure_lookup_table(); seen = set(); filled = 0
    for e in bm.edges:
        if not e.is_boundary or e.index in seen: continue
        loop, stack, L, pts = [], [e], 0.0, []
        while stack:
            x = stack.pop()
            if x.index in seen: continue
            seen.add(x.index); loop.append(x); L += x.calc_length(); pts += [tuple(vv.co) for vv in x.verts]
            for vv in x.verts:
                for y in vv.link_edges:
                    if y.is_boundary and y.index not in seen: stack.append(y)
        # small holes, and the long but narrow slivers left open along the panel/rim seam
        ext = np.sort(np.ptp(np.array(pts), 0))
        if L < maxlen or ext[1] < float(os.environ.get('SLIVER', 0.045)):
            f = bmesh.ops.holes_fill(bm, edges=loop, sides=0)['faces']
            if f: filled += len(bmesh.ops.triangulate(bm, faces=f, quad_method='BEAUTY', ngon_method='BEAUTY')['faces'])
    bm.to_mesh(o.data); bm.free(); o.data.update(); return filled
def zip_seams(o, dist=float(os.environ.get('ZIP', 0.004))):
    # weld the open edges on both sides of a seam gap together (only vertices on open edges move)
    bm = bmesh.new(); bm.from_mesh(o.data); bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bv = [v for v in bm.verts if any(e.is_boundary for e in v.link_edges)]; n0 = len(bv)
    bmesh.ops.remove_doubles(bm, verts=bv, dist=dist)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=1e-6)
    bm.to_mesh(o.data); bm.free(); o.data.update(); return n0
for i in ((0, 1, 2) if os.environ.get('ZIP_SEAMS', '0') == '1' else ()):  # off: did not close the slots
    print('part', i, 'seam zipped:', zip_seams(P[i]), 'open-edge vertices')
for i in (0, 1, 2, 80):
    print('part', i, 'pin-holes capped with', fill_small_holes(P[i]), 'faces')

# slots: narrow through-slots in the rims (where a spine bracket tab used to sit) show the backdrop straight
# through. Each remaining open edge chain that is thin in one direction gets a small black block filling it
def plug_slots(o, thin=float(os.environ.get('SLOT_THIN', 0.007)), big=float(os.environ.get('SLOT_MAX', 0.1))):
    bm = bmesh.new(); bm.from_mesh(o.data); bm.edges.ensure_lookup_table(); seen = set(); boxes = []
    for e in bm.edges:
        if not e.is_boundary or e.index in seen: continue
        stack, pts = [e], []
        while stack:
            x = stack.pop()
            if x.index in seen: continue
            seen.add(x.index); pts += [tuple(vv.co) for vv in x.verts]
            for vv in x.verts:
                for y in vv.link_edges:
                    if y.is_boundary and y.index not in seen: stack.append(y)
        p = np.array(pts); lo, hi = p.min(0), p.max(0); ext = hi - lo
        if ext.min() < thin and ext.max() < big and ext.max() > 0.004: boxes.append((lo, hi))
    for lo, hi in boxes:
        pad = 0.0008; c = (lo + hi) / 2; sz = hi - lo + 2 * pad
        r = bmesh.ops.create_cube(bm, size=1.0)
        for vv in r['verts']: vv.co = mathutils.Vector(c) + mathutils.Vector((vv.co.x * sz[0], vv.co.y * sz[1], vv.co.z * sz[2]))
        for f in {f for vv in r['verts'] for f in vv.link_faces}: f.material_index = 0
    bm.to_mesh(o.data); bm.free(); o.data.update(); return len(boxes)
for i in ((0, 1, 2) if os.environ.get('PLUGS', '0') == '1' else ()):     # off: blocks showed on the rims
    print('part', i, 'slots plugged:', plug_slots(P[i]))

# seam gaps: where the closed panel meets the original rim, slivers of backdrop still show through. With the spine
# split off, backrest and lumbar are compact shells: seal each one (fine voxel remesh + a 1 mm closing) and carry the
# panel/rim material split over from the original surface
def seal(o, grow=float(os.environ.get('SEAL_GROW', 0.0012)), vox=float(os.environ.get('SEAL_VOX', 0.0008))):
    mi = np.zeros(len(o.data.polygons), np.int32); o.data.polygons.foreach_get('material_index', mi); c0 = face_centres(o)
    for step in ('remesh', 'grow', 'remesh', 'shrink', 'smooth'):
        if step in ('grow', 'shrink'):
            md = o.modifiers.new(step, 'DISPLACE'); md.mid_level = 0.0; md.strength = grow if step == 'grow' else -grow; md.direction = 'NORMAL'
        elif step == 'remesh':
            md = o.modifiers.new(step, 'REMESH'); md.mode = 'VOXEL'; md.voxel_size = vox
        else:
            md = o.modifiers.new(step, 'CORRECTIVE_SMOOTH'); md.iterations = 4; md.use_only_smooth = True
        apply(o, md)
    _, k = cKDTree(c0).query(face_centres(o)); o.data.polygons.foreach_set('material_index', mi[k]); o.data.update()
    return len(o.data.polygons)
for i in ((0, 1) if os.environ.get('SEAL', '0') == '1' else ()):     # off: Tripo's surface is not watertight enough
    print('part', i, 'sealed:', seal(P[i]), 'faces')

# ---- materials: Tripo's maps carry painted shadows, smudges and a blotchy normal map; everything gets a clean
# material in the real chair's colours (calibrated against the real photos with anim3's LIGHT=0.22)
def mat(name, colour, rough, spec=0.35, sheen=0.0, metal=0.0):
    m = bpy.data.materials.get(name)
    if m: return m
    m = bpy.data.materials.new(name); m.use_nodes = True; pb = m.node_tree.nodes['Principled BSDF']
    pb.inputs['Base Color'].default_value = (*colour, 1); pb.inputs['Roughness'].default_value = rough
    pb.inputs['Specular IOR Level'].default_value = spec; pb.inputs['Metallic'].default_value = metal
    pb.inputs['Sheen Weight'].default_value = sheen; pb.inputs['Sheen Roughness'].default_value = 0.5
    return m

PL = tuple(float(x) for x in os.environ.get('PLASTIC_RGB', '0.012,0.012,0.013').split(','))
plastic = mat('frame_plastic', PL, 0.5, spec=0.12)
seat = mat('seat_fabric', (0.008, 0.008, 0.009), 0.95, spec=0.04, sheen=0.0)
rubber = mat('rubber', (0.012, 0.012, 0.013), 0.55)
pad = mat('arm_pad', (0.012, 0.012, 0.013), 0.78, spec=0.12)
chrome = mat('chrome', (0.86, 0.86, 0.88), float(os.environ.get('CHROME_ROUGH', 0.14)), metal=1.0)
rimf = mat('headrest_rim_fabric', (0.012, 0.012, 0.013), 0.92, spec=0.08)

# the real mesh fabric: a seamless tile cut from the real chair's photo, true to its colours (no grading),
# mapped flat from the front at the real stripe size; a touch of sheen like the real woven mesh
fab = bpy.data.materials.new('mesh_fabric'); fab.use_nodes = True; nt = fab.node_tree; pb = nt.nodes['Principled BSDF']
pb.inputs['Roughness'].default_value = 0.85; pb.inputs['Specular IOR Level'].default_value = 0.2
pb.inputs['Sheen Weight'].default_value = 0.25
t = nt.nodes.new('ShaderNodeTexImage'); t.image = bpy.data.images.load(TILE); t.extension = 'REPEAT'
u = nt.nodes.new('ShaderNodeUVMap'); u.uv_map = 'mesh_uv'; nt.links.new(u.outputs[0], t.inputs['Vector'])
nt.links.new(t.outputs['Color'], pb.inputs['Base Color'])
bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = 0.15; bp.inputs['Distance'].default_value = 0.0004
nt.links.new(t.outputs['Color'], bp.inputs['Height']); nt.links.new(bp.outputs['Normal'], pb.inputs['Normal'])
# the real mesh is a see-through weave: the V frame and brackets behind it show faintly (MESH_SEE of the light passes)
see = float(os.environ.get('MESH_SEE', 0.3))
if see > 0:
    tr = nt.nodes.new('ShaderNodeBsdfTransparent'); mx = nt.nodes.new('ShaderNodeMixShader'); mx.inputs[0].default_value = see
    mo = nt.nodes['Material Output']; nt.links.new(pb.outputs[0], mx.inputs[1]); nt.links.new(tr.outputs[0], mx.inputs[2]); nt.links.new(mx.outputs[0], mo.inputs['Surface'])
TW, TH = (float(x) for x in os.environ.get('TILE_SIZE', '0.194,0.176').split(','))

def set_all(o, m):
    o.data.materials.clear(); o.data.materials.append(m)

def mesh_uv(o):
    v = verts(o); lv = np.zeros(len(o.data.loops), int); o.data.loops.foreach_get('vertex_index', lv); p = v[lv]
    o.data.uv_layers.new(name='mesh_uv').data.foreach_set('uv', np.c_[-p[:, 0] / TW, p[:, 2] / TH].astype(np.float32).reshape(-1))

for i, rest in ((0, plastic), (1, plastic), (2, rimf)):
    o = P[i]; k = len(o.data.materials) - 1
    mi = np.zeros(len(o.data.polygons), np.int32); o.data.polygons.foreach_get('material_index', mi)
    o.data.materials.clear(); o.data.materials.append(rest); o.data.materials.append(fab)
    o.data.polygons.foreach_set('material_index', (mi == k).astype(np.int32)); mesh_uv(o); o.data.update()
for i in (80, 12, 14, 7, 13, 17): set_all(P[i], plastic)
# the real chair's chrome band round the spine where the V meets the lower bracket (Tripo painted it silver here)
RZ0, RZ1 = (float(x) for x in os.environ.get('RING', '0.4705,0.4762').split(','))
fr = P[80]; bm = bmesh.new(); bm.from_mesh(fr.data)
for zc in (RZ0, RZ1):
    bmesh.ops.bisect_plane(bm, geom=list(bm.verts) + list(bm.edges) + list(bm.faces), plane_co=(0, 0, zc), plane_no=(0, 0, 1))
bm.to_mesh(fr.data); bm.free(); fr.data.update(); fr.data.materials.append(chrome)
c = face_centres(fr); ring = (c[:, 2] > RZ0) & (c[:, 2] < RZ1)
fr.data.polygons.foreach_set('material_index', ring.astype(np.int32)); fr.data.update()
print('chrome band on the spine:', int(ring.sum()), 'faces')
set_all(P[3], seat)
for i in GROUPS['wheels']: set_all(P[i], rubber)
for i in (5, 6):                              # armrests: soft-touch pad on top
    o = P[i]; set_all(o, plastic); o.data.materials.append(pad); c = face_centres(o)
    o.data.polygons.foreach_set('material_index', (c[:, 2] > c[:, 2].max() - 0.028).astype(np.int32)); o.data.update()
# base: like the real chair, polished chrome legs and a black hub (BASE=black keeps it all black)
b = P[4]; set_all(b, plastic)
if os.environ.get('BASE', 'chrome') == 'chrome':
    b.data.materials.append(chrome); v = verts(b); top = v[v[:, 2] > v[:, 2].max() - 0.03]; ax = (top[:, :2].min(0) + top[:, :2].max(0)) / 2
    c = face_centres(b); legs = np.hypot(c[:, 0] - ax[0], c[:, 1] - ax[1]) > float(os.environ.get('HUB_R', 0.055))
    b.data.polygons.foreach_set('material_index', legs.astype(np.int32)); b.data.update()
    print('chrome legs:', int(legs.sum()), 'faces; hub at', ax.round(3))
for o in P.values():
    o.data.polygons.foreach_set('use_smooth', np.ones(len(o.data.polygons), bool)); o.data.update()

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
for im in list(bpy.data.images):
    if im.users == 0: bpy.data.images.remove(im)
bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=OUT)
print('saved', OUT, sorted(o.name for o in bpy.data.objects if o.type == 'MESH'))
