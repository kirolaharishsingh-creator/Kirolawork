# True-3D exploded view of the Tripo chair model in Blender (Cycles, CPU).
# Usage (inside the sandbox, next to chair.glb, co.npy, lab.npy, parts.py):
#   python3 anim.py test   -> renders a few check frames
#   python3 anim.py full   -> renders every frame to frames/####.png (RGBA, shadow-catcher floor)
import bpy, sys, math, time, numpy as np, mathutils
from parts import vertex_parts, PART_NAMES

MODE = sys.argv[-1]
FPS, N = 24, 115                  # 4.8 s
import os
W = int(os.environ.get('RES', 960)); H = W * 9 // 16
SAMPLES = 16

# part -> (stagger order, offset in metres: x forward, y lateral, z up)
MOVES = {
    'headrest':  (0, (-0.05, 0.0, 0.22)),
    'backrest':  (1, (-0.10, 0.0, 0.12)),
    'frame':     (1, (-0.30, 0.0, 0.02)),
    'lumbar':    (2, (-0.15, 0.0, 0.00)),
    'seat':      (3, (0.00, 0.0, 0.14)),
    'arm_r':     (4, (0.00, 0.17, 0.06)),
    'arm_l':     (4, (0.00, -0.17, 0.06)),
    'mechanism': (5, (0.00, 0.0, -0.04)),
    'gas_lift':  (5, (0.00, 0.0, -0.10)),
    'base':      (6, (0.00, 0.0, -0.18)),
}
WHEEL_DROP, WHEEL_OUT = 0.26, 0.06
T0, STEP, DUR = 0.3, 0.15, 0.6    # explode: part i starts at T0 + i*STEP and takes DUR
HOLD_END = 2.6                    # return starts here, in reverse order
FLOOR0 = -0.4904

def smooth(u):
    u = min(max(u, 0.0), 1.0); return u * u * (3 - 2 * u)

def amount(i, t):
    out = smooth((t - (T0 + i * STEP)) / DUR)
    back = smooth((t - (HOLD_END + (7 - i) * STEP)) / DUR)
    return out * (1 - back)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath='chair.glb')
src = [o for o in bpy.context.scene.objects if o.type == 'MESH'][0]
mat = src.data.materials[0]

# label faces by part (via loose-shell labels), then split into one object per part
co = np.load('co.npy'); lab = np.load('lab.npy')
vpart = vertex_parts(co, lab, np.load('tri.npy'))
me = src.data
nf = len(me.polygons); lst = np.zeros(nf, int); me.polygons.foreach_get('loop_start', lst)
lv = np.zeros(len(me.loops), int); me.loops.foreach_get('vertex_index', lv)
fpart = vpart[lv[lst]]
lt = np.zeros(nf, int); me.polygons.foreach_get('loop_total', lt)
assert (lt == 3).all(), 'expects a triangle mesh'
li = lst[:, None] + np.arange(3)                    # loop indices of each triangle
tri = lv[li]
uv = np.zeros(len(me.loops) * 2, np.float32); me.uv_layers[0].data.foreach_get('uv', uv); uv = uv.reshape(-1, 2)[li]
import bmesh
cap_mat = bpy.data.materials.new('cap'); cap_mat.use_nodes = True
cp = cap_mat.node_tree.nodes['Principled BSDF']; cp.inputs['Base Color'].default_value = (0.012, 0.012, 0.013, 1)
cp.inputs['Roughness'].default_value = 1.0; cp.inputs['Specular IOR Level'].default_value = 0.0   # no sheen: caps read as dark openings

def cap_holes(m, min_faces=int(os.environ.get('MINF', 800))):
    # weld the AI model's texture seams, drop small loose shards, then close every cut opening with a matte black cap
    bm = bmesh.new(); bm.from_mesh(m)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=2e-4)
    bm.to_mesh(m); bm.free()
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    nf_ = len(m.polygons); lt_ = np.zeros(nf_, int); m.polygons.foreach_get('loop_total', lt_)
    ls_ = np.zeros(nf_, int); m.polygons.foreach_get('loop_start', ls_)
    lv_ = np.zeros(len(m.loops), int); m.loops.foreach_get('vertex_index', lv_)
    fi = np.repeat(np.arange(nf_), lt_); nv_ = len(m.vertices)
    # faces sharing a vertex are connected: bipartite face-vertex graph
    G = coo_matrix((np.ones(len(lv_)), (fi, nf_ + lv_)), shape=(nf_ + nv_, nf_ + nv_))
    _, comp = connected_components(G, directed=False)
    fc = comp[:nf_]; size = np.bincount(fc)
    drop = np.nonzero(size[fc] < min_faces)[0]
    bm = bmesh.new(); bm.from_mesh(m); bm.faces.ensure_lookup_table()
    if len(drop):
        bmesh.ops.delete(bm, geom=[bm.faces[i] for i in drop], context='FACES')
        loose = [v for v in bm.verts if not v.link_faces]
        if loose: bmesh.ops.delete(bm, geom=loose, context='VERTS')
    edges = [e for e in bm.edges if e.is_boundary]
    new = bmesh.ops.holes_fill(bm, edges=edges, sides=0)['faces']
    if new:
        tri_ = bmesh.ops.triangulate(bm, faces=new)['faces']
        for f in tri_: f.material_index = 1; f.smooth = False
        bmesh.ops.recalc_face_normals(bm, faces=tri_)
    bm.to_mesh(m); bm.free(); m.update()
    print('cleaned', m.name, 'dropped faces', len(drop), flush=True)
    return len(new)

objs = {}
for p, name in enumerate(PART_NAMES):
    sel = np.nonzero(fpart == p)[0]
    t = tri[sel]; used, inv = np.unique(t, return_inverse=True)
    m2 = bpy.data.meshes.new(name)
    m2.vertices.add(len(used)); m2.vertices.foreach_set('co', co[used].reshape(-1))       # world coordinates; objects keep identity transforms
    m2.loops.add(len(sel) * 3); m2.loops.foreach_set('vertex_index', inv.reshape(-1).astype(np.int32))
    m2.polygons.add(len(sel)); m2.polygons.foreach_set('loop_start', (np.arange(len(sel)) * 3).astype(np.int32))
    m2.uv_layers.new(); m2.uv_layers[0].data.foreach_set('uv', uv[sel].reshape(-1))
    m2.update(); m2.shade_smooth(); m2.materials.append(mat); m2.materials.append(cap_mat)
    o = bpy.data.objects.new(name, m2); bpy.context.scene.collection.objects.link(o); objs[name] = o
bpy.data.objects.remove(src)

def mesh_arrays(m):
    nv = len(m.vertices); co_ = np.zeros(nv * 3, np.float32); m.vertices.foreach_get('co', co_)
    nl = len(m.loops); lv_ = np.zeros(nl, np.int32); m.loops.foreach_get('vertex_index', lv_)
    nf = len(m.polygons); ls_ = np.zeros(nf, np.int32); m.polygons.foreach_get('loop_start', ls_)
    lt_ = np.zeros(nf, np.int32); m.polygons.foreach_get('loop_total', lt_)
    uv_ = np.zeros(nl * 2, np.float32); m.uv_layers[0].data.foreach_get('uv', uv_)
    return co_.reshape(-1, 3), lv_, ls_, lt_, uv_.reshape(-1, 2)

def mesh_from_arrays(name, co_, lv_, ls_, mi_, uv_):
    m = bpy.data.meshes.new(name)
    m.vertices.add(len(co_)); m.vertices.foreach_set('co', co_.reshape(-1))
    m.loops.add(len(lv_)); m.loops.foreach_set('vertex_index', lv_)
    m.polygons.add(len(ls_)); m.polygons.foreach_set('loop_start', ls_)
    m.uv_layers.new(); m.uv_layers[0].data.foreach_set('uv', uv_.reshape(-1))
    m.polygons.foreach_set('material_index', mi_); m.update(); return m

def plane_split(a, b, z, region, claim_above=None, outside_to_b=False):
    # re-divide parts a (above) and b (below) along the flat plane at height z, inside region(x, y):
    # a clean straight cut instead of the AI model's ragged patchwork seam
    A, B = mesh_arrays(objs[a].data), mesh_arrays(objs[b].data)
    nva, nla = len(A[0]), len(A[1])
    joined = mesh_from_arrays('join', np.concatenate([A[0], B[0]]), np.concatenate([A[1], B[1] + nva]),
                              np.concatenate([A[2], B[2] + nla]),
                              np.concatenate([np.zeros(len(A[2]), np.int32), np.ones(len(B[2]), np.int32)]),
                              np.concatenate([A[4], B[4]]))
    bm = bmesh.new(); bm.from_mesh(joined); bpy.data.meshes.remove(joined)
    geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
    bmesh.ops.bisect_plane(bm, geom=geom, plane_co=(0, 0, z), plane_no=(0, 0, 1))
    for f in bm.faces:
        c = f.calc_center_median()
        if region(c.x, c.y): f.material_index = 0 if c.z > z else 1
        elif claim_above is not None and c.z > claim_above: f.material_index = 0   # outside the footprint, high pieces belong to a
        elif outside_to_b and f.material_index == 0: f.material_index = 1          # a keeps only what is inside the region
    for name, keep in ((a, 0), (b, 1)):
        bm2 = bm.copy()
        bmesh.ops.delete(bm2, geom=[f for f in bm2.faces if f.material_index != keep], context='FACES')
        bmesh.ops.delete(bm2, geom=[v for v in bm2.verts if not v.link_faces], context='VERTS')
        for f in bm2.faces: f.material_index = 0
        bm2.to_mesh(objs[name].data); bm2.free(); objs[name].data.update()
    bm.free()
    print('split', a, b, z, len(objs[a].data.polygons), len(objs[b].data.polygons), flush=True)

_g = objs['gas_lift'].data; _gv = np.zeros(len(_g.vertices) * 3); _g.vertices.foreach_get('co', _gv); _gv = _gv.reshape(-1, 3)
_col = _gv[(_gv[:, 2] > -0.32) & (_gv[:, 2] < -0.27)]
HUBXY = tuple(((_col[:, :2].min(0) + _col[:, :2].max(0)) / 2).tolist()) if len(_col) else (0.01, 0.0)
print('hub centre', HUBXY, flush=True)
near_hub = lambda x, y: math.hypot(x - HUBXY[0], y - HUBXY[1]) < 0.07
if os.environ.get('PLANES', '1') == '1':
    # only under the mechanism's own footprint, so the seat keeps its whole underside
    plane_split('seat', 'mechanism', -0.172, lambda x, y: -0.08 < x < 0.16 and abs(y) < 0.1, claim_above=-0.205)
    plane_split('mechanism', 'gas_lift', -0.255, lambda x, y: math.hypot(x - HUBXY[0], y - HUBXY[1]) < 0.045)
    # the column only: the hub socket and arm roots stay with the base
    # the chrome rod only: its wider housing stays with the base hub
    plane_split('gas_lift', 'base', -0.33, lambda x, y: math.hypot(x - HUBXY[0], y - HUBXY[1]) < 0.045, outside_to_b=True)
    # armrests end in a clean cut where their bracket meets the seat
    plane_split('arm_r', 'seat', -0.16, lambda x, y: y > 0.21)       # outside the cushion's edge only
    plane_split('arm_l', 'seat', -0.16, lambda x, y: y < -0.21)
    for i in range(5):
        o = objs['wheel%d' % i]; v = np.zeros(len(o.data.vertices) * 3); o.data.vertices.foreach_get('co', v); v = v.reshape(-1, 3)
        wx, wy = v[:, 0].mean(), v[:, 1].mean()
        plane_split('base', 'wheel%d' % i, -0.418, lambda x, y, wx=wx, wy=wy: math.hypot(x - wx, y - wy) < 0.05)
if os.environ.get('CAPS', '1') == '1':
    for o in objs.values(): cap_holes(o.data)
print('parts', sorted(objs), flush=True)

# the AI texture reads mid-grey; darken it back to the real black mesh/plastic
nt = mat.node_tree; pb = [n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'][0]
if pb.inputs['Base Color'].links:
    lk = pb.inputs['Base Color'].links[0]; g = nt.nodes.new('ShaderNodeGamma'); g.inputs[1].default_value = 2.2
    nt.links.new(lk.from_socket, g.inputs[0]); nt.links.new(g.outputs[0], pb.inputs['Base Color'])
# cap the texture's brightness: the AI texture has light blotches in the mesh fabric that read as patches
sepc = nt.nodes.new('ShaderNodeSeparateColor'); comb = nt.nodes.new('ShaderNodeCombineColor')
src_sock = pb.inputs['Base Color'].links[0].from_socket
nt.links.new(src_sock, sepc.inputs[0])
for ch in range(3):
    mn = nt.nodes.new('ShaderNodeMath'); mn.operation = 'MINIMUM'; mn.inputs[1].default_value = 0.045
    nt.links.new(sepc.outputs[ch], mn.inputs[0]); nt.links.new(mn.outputs[0], comb.inputs[ch])
nt.links.new(comb.outputs[0], pb.inputs['Base Color'])
# black plastic/fabric: no metal, soft low sheen (the ORM map's metallic made it read grey)
chrome_src = mat.copy()
for l in list(pb.inputs['Metallic'].links): nt.links.remove(l)
pb.inputs['Metallic'].default_value = 0.0
pb.inputs['Specular IOR Level'].default_value = 0.25
# roughness floor: glossy spots in the AI roughness map mirrored the bright ceiling as white patches
rl = pb.inputs['Roughness'].links[0].from_socket
rmax = nt.nodes.new('ShaderNodeMath'); rmax.operation = 'MAXIMUM'; rmax.inputs[1].default_value = 0.6
nt.links.new(rl, rmax.inputs[0]); nt.links.new(rmax.outputs[0], pb.inputs['Roughness'])

# polished chrome for the base and gas lift (keeps the texture's dark details)
chrome = chrome_src; chrome.name = 'chrome'
bsdf = [n for n in chrome.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'][0]
for l in list(bsdf.inputs['Metallic'].links) + list(bsdf.inputs['Roughness'].links) + list(bsdf.inputs['Base Color'].links):
    chrome.node_tree.links.remove(l)
bsdf.inputs['Base Color'].default_value = (0.85, 0.85, 0.88, 1)
bsdf.inputs['Metallic'].default_value = 1.0; bsdf.inputs['Roughness'].default_value = 0.18
for n in ('base',):
    objs[n].data.materials[0] = chrome

def fabric_material(src_mat):
    f = src_mat.copy(); f.name = 'backrest_fabric'; nt = f.node_tree
    pb = [n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'][0]
    col = pb.inputs['Base Color'].links[0].from_socket
    mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'; mix.inputs['Factor'].default_value = 0.92
    nt.links.new(col, mix.inputs[6]); mix.inputs[7].default_value = (0.04, 0.04, 0.043, 1)
    tc = nt.nodes.new('ShaderNodeTexCoord'); wave = nt.nodes.new('ShaderNodeTexWave')
    wave.bands_direction = 'Z'; wave.inputs['Scale'].default_value = 140.0; wave.inputs['Distortion'].default_value = 0.0
    nt.links.new(tc.outputs['Object'], wave.inputs['Vector'])
    ramp = nt.nodes.new('ShaderNodeMapRange'); ramp.inputs['To Min'].default_value = 0.35; ramp.inputs['To Max'].default_value = 1.0
    nt.links.new(wave.outputs['Fac'], ramp.inputs['Value'])
    mul = nt.nodes.new('ShaderNodeMix'); mul.data_type = 'RGBA'; mul.blend_type = 'MULTIPLY'; mul.inputs['Factor'].default_value = 1.0
    nt.links.new(mix.outputs[2], mul.inputs[6]); nt.links.new(ramp.outputs['Result'], mul.inputs[7])
    nt.links.new(mul.outputs[2], pb.inputs['Base Color'])
    return f

def black_backfaces(m):
    # the inside of a cut opening shows the model's back faces: make them flat black so openings read as dark recesses
    nt = m.node_tree; out = [n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'][0]
    surf = out.inputs['Surface'].links[0].from_socket
    geo = nt.nodes.new('ShaderNodeNewGeometry'); dark = nt.nodes.new('ShaderNodeBsdfDiffuse')
    dark.inputs['Color'].default_value = (0.004, 0.004, 0.004, 1)
    mix = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(geo.outputs['Backfacing'], mix.inputs[0]); nt.links.new(surf, mix.inputs[1]); nt.links.new(dark.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs['Surface'])
for m in (mat, chrome): black_backfaces(m)
fabric = fabric_material(mat); objs['backrest'].data.materials[0] = fabric

def treat_black(m):
    # same treatment as the chair's own material: real black, no metal, soft sheen, capped brightness
    nt = m.node_tree; pb = [n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'][0]
    if pb.inputs['Base Color'].links:
        src = pb.inputs['Base Color'].links[0].from_socket
        g = nt.nodes.new('ShaderNodeGamma'); g.inputs[1].default_value = 2.2; nt.links.new(src, g.inputs[0])
        sep = nt.nodes.new('ShaderNodeSeparateColor'); cmb = nt.nodes.new('ShaderNodeCombineColor'); nt.links.new(g.outputs[0], sep.inputs[0])
        for ch in range(3):
            mn = nt.nodes.new('ShaderNodeMath'); mn.operation = 'MINIMUM'; mn.inputs[1].default_value = 0.06
            nt.links.new(sep.outputs[ch], mn.inputs[0]); nt.links.new(mn.outputs[0], cmb.inputs[ch])
        nt.links.new(cmb.outputs[0], pb.inputs['Base Color'])
    for l in list(pb.inputs['Metallic'].links): nt.links.remove(l)
    pb.inputs['Metallic'].default_value = 0.0; pb.inputs['Specular IOR Level'].default_value = 0.25
    if pb.inputs['Roughness'].links:
        rl = pb.inputs['Roughness'].links[0].from_socket
        rm = nt.nodes.new('ShaderNodeMath'); rm.operation = 'MAXIMUM'; rm.inputs[1].default_value = 0.6
        nt.links.new(rl, rm.inputs[0]); nt.links.new(rm.outputs[0], pb.inputs['Roughness'])

PROJBACK = os.environ.get('PROJBACK')       # 'front.png,back.png': isolated real backrest photos
if PROJBACK:
    # keep the original backrest shape (it fits the chair), flatten the V dents in its mesh area,
    # and wrap the real fabric from the isolated photos onto it: front photo on the front, back photo on the back
    ob = objs['backrest']; me = ob.data
    bm = bmesh.new(); bm.from_mesh(me)
    inner = [v for v in bm.verts if abs(v.co.y) < 0.135 and 0.125 < v.co.z < 0.355]
    for _ in range(int(os.environ.get('SMOOTH', 40))):
        bmesh.ops.smooth_vert(bm, verts=inner, factor=0.5, use_axis_x=True, use_axis_y=False, use_axis_z=False)
    bm.to_mesh(me); bm.free(); me.update()
    v = np.zeros(len(me.vertices) * 3); me.vertices.foreach_get('co', v); v = v.reshape(-1, 3)
    y0, y1 = np.percentile(v[:, 1], [0.5, 99.5]); z0, z1 = np.percentile(v[:, 2], [0.5, 99.5])
    def panel_box(img):
        w, h = img.size; px = np.array(img.pixels[:], np.float32).reshape(h, w, 4)
        m = px[..., :3].mean(-1) < 0.6; ys, xs = np.nonzero(m)
        return xs.min() / w, xs.max() / w, ys.min() / h, ys.max() / h      # pixel rows run bottom to top in Blender
    imgs = [bpy.data.images.load(os.path.abspath(p)) for p in PROJBACK.split(',')]
    boxes = [panel_box(im) for im in imgs]
    lv = np.zeros(len(me.loops), np.int32); me.loops.foreach_get('vertex_index', lv)
    P = v[lv]; s = (P[:, 1] - y0) / (y1 - y0); t = (P[:, 2] - z0) / (z1 - z0)
    for name, (u0, u1, w0, w1), mirror in (('front_uv', boxes[0], False), ('back_uv', boxes[1], True)):
        lay = me.uv_layers.new(name=name)
        su = (1 - s) if mirror else s          # seen from behind, left and right swap
        # front view: chair's +y appears on the right of the photo
        uv = np.c_[u0 + su * (u1 - u0), w0 + t * (w1 - w0)]
        lay.data.foreach_set('uv', uv.astype(np.float32).reshape(-1))
    pm = bpy.data.materials.new('backrest_photo'); pm.use_nodes = True; nt = pm.node_tree
    pb = nt.nodes['Principled BSDF']; pb.inputs['Roughness'].default_value = 0.7; pb.inputs['Specular IOR Level'].default_value = 0.25
    texs = []
    for im, uvname in zip(imgs, ('front_uv', 'back_uv')):
        tx = nt.nodes.new('ShaderNodeTexImage'); tx.image = im; tx.extension = 'EXTEND'
        uvn = nt.nodes.new('ShaderNodeUVMap'); uvn.uv_map = uvname; nt.links.new(uvn.outputs[0], tx.inputs['Vector']); texs.append(tx)
    geo = nt.nodes.new('ShaderNodeNewGeometry'); sepn = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(geo.outputs['Normal'], sepn.inputs[0])
    gt = nt.nodes.new('ShaderNodeMath'); gt.operation = 'GREATER_THAN'; gt.inputs[1].default_value = 0.0; nt.links.new(sepn.outputs['X'], gt.inputs[0])
    mx = nt.nodes.new('ShaderNodeMix'); mx.data_type = 'RGBA'; nt.links.new(gt.outputs[0], mx.inputs['Factor'])
    nt.links.new(texs[1].outputs['Color'], mx.inputs[6]); nt.links.new(texs[0].outputs['Color'], mx.inputs[7])
    # photo colours include the studio light already: bring them to surface colour for re-lighting
    gm = nt.nodes.new('ShaderNodeGamma'); gm.inputs[1].default_value = 1.6; nt.links.new(mx.outputs[2], gm.inputs[0])
    br = nt.nodes.new('ShaderNodeBrightContrast'); br.inputs['Bright'].default_value = 0.0; nt.links.new(gm.outputs[0], br.inputs[0])
    nt.links.new(br.outputs[0], pb.inputs['Base Color'])
    me.materials.clear(); me.materials.append(pm); me.materials.append(cap_mat)
    fabric = pm
    print('backrest re-skinned from photos', boxes, flush=True)

NEWBACK = os.environ.get('NEWBACK')
if NEWBACK:
    # swap in the backrest rebuilt from the real photos, fitted to where the old one sat
    from scipy.spatial import cKDTree
    old = objs['backrest']; ov = np.zeros(len(old.data.vertices) * 3); old.data.vertices.foreach_get('co', ov); ov = ov.reshape(-1, 3)
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=NEWBACK)
    nb = [o for o in bpy.data.objects if o not in before and o.type == 'MESH'][0]
    nm = nb.data; nm.transform(nb.matrix_world); nb.matrix_world = mathutils.Matrix.Identity(4)
    nv = np.zeros(len(nm.vertices) * 3); nm.vertices.foreach_get('co', nv); nv = nv.reshape(-1, 3)
    ext = nv.max(0) - nv.min(0); print('new backrest extents xyz', ext.round(3), flush=True)
    FACE = os.environ.get('NEWBACK_FACE')            # e.g. '+x': Tripo built a box; keep only that clean face
    if FACE:
        ax = 'xyz'.index(FACE[1]); sgn = 1 if FACE[0] == '+' else -1
        d = float(os.environ.get('NEWBACK_DEPTH', 0.08)) * ext[ax]
        edge = nv[:, ax].max() - d if sgn > 0 else nv[:, ax].min() + d
        keepv = (nv[:, ax] > edge) if sgn > 0 else (nv[:, ax] < edge)
        bm = bmesh.new(); bm.from_mesh(nm); bm.verts.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[f for f in bm.faces if not all(keepv[v.index] for v in f.verts)], context='FACES')
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
        bm.to_mesh(nm); bm.free(); nm.update()
        nv = np.zeros(len(nm.vertices) * 3); nm.vertices.foreach_get('co', nv); nv = nv.reshape(-1, 3)
        ext = nv.max(0) - nv.min(0); print('kept face', FACE, 'extents', ext.round(3), flush=True)
    thin = int(np.argmin(ext)); horiz = [i for i in (0, 1) if i != thin][0] if thin != 2 else 1
    # local frame of the new panel: u = across, w = up, t = thickness (sign: +t is the panel's front)
    u, w, t = nv[:, horiz], nv[:, 2], nv[:, thin]
    if os.environ.get('NEWBACK_FLIPT') == '1': t = -t
    if os.environ.get('NEWBACK_FLIPU') == '1': u = -u
    # target: old backrest across = y, up = z; front faces +x
    oy0, oy1 = np.percentile(ov[:, 1], [0.5, 99.5]); oz0, oz1 = np.percentile(ov[:, 2], [0.5, 99.5])
    su = (oy1 - oy0) / (u.max() - u.min()); sw = (oz1 - oz0) / (w.max() - w.min()); s = (su + sw) / 2
    Y = oy0 + (u - u.min()) * su; Z = oz0 + (w - w.min()) * sw
    # bend the flat panel onto the old backrest's curved mid-surface (its side profile)
    tree = cKDTree(ov[:, 1:3]); _, idx = tree.query(np.c_[Y, Z], k=24)
    mid = np.median(ov[idx, 0], axis=1)
    X = mid + (t - (t.max() + t.min()) / 2) * s
    nm.vertices.foreach_set('co', np.c_[X, Y, Z].reshape(-1)); nm.update()
    for p in nm.polygons: p.use_smooth = True
    for m in nm.materials:
        treat_black(m)
        if not FACE: black_backfaces(m)
    bpy.data.objects.remove(old); nb.name = 'backrest'; objs['backrest'] = nb
    print('new backrest fitted, scale', round(s, 4), flush=True)

def add_glow(m):
    nt = m.node_tree; out = [n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'][0]
    surf = out.inputs['Surface'].links[0].from_socket
    lw = nt.nodes.new('ShaderNodeLayerWeight'); lw.inputs['Blend'].default_value = 0.2
    pw = nt.nodes.new('ShaderNodeMath'); pw.operation = 'POWER'; pw.inputs[1].default_value = 4.0
    at = nt.nodes.new('ShaderNodeAttribute'); at.attribute_type = 'OBJECT'; at.attribute_name = 'glow'
    mu = nt.nodes.new('ShaderNodeMath'); mu.operation = 'MULTIPLY'
    mu2 = nt.nodes.new('ShaderNodeMath'); mu2.operation = 'MULTIPLY'; mu2.inputs[1].default_value = 7.0
    em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Color'].default_value = (0.82, 0.92, 1.0, 1)
    add = nt.nodes.new('ShaderNodeAddShader')
    L = nt.links
    L.new(lw.outputs['Facing'], pw.inputs[0]); L.new(pw.outputs[0], mu.inputs[0]); L.new(at.outputs['Fac'], mu.inputs[1])
    L.new(mu.outputs[0], mu2.inputs[0]); L.new(mu2.outputs[0], em.inputs['Strength'])
    L.new(surf, add.inputs[0]); L.new(em.outputs[0], add.inputs[1]); L.new(add.outputs[0], out.inputs['Surface'])
for m in {mat, chrome, cap_mat, fabric, *objs['backrest'].data.materials}: add_glow(m)

# wheel offsets: straight down plus outward from the hub
hub = np.array([0.01, 0.0])
for i in range(5):
    o = objs['wheel%d' % i]
    v = np.zeros(len(o.data.vertices) * 3); o.data.vertices.foreach_get('co', v); v = v.reshape(-1, 3)
    d = v[:, :2].mean(0) - hub; d /= np.linalg.norm(d)
    MOVES['wheel%d' % i] = (7, (WHEEL_OUT * d[0], WHEEL_OUT * d[1], -WHEEL_DROP))

# scene: camera, lights, shadow-catcher floor that follows the wheels
sc = bpy.context.scene
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = SAMPLES
sc.cycles.use_denoising = True; sc.render.film_transparent = True
sc.render.resolution_x, sc.render.resolution_y = W, H; sc.render.fps = FPS
sc.render.use_persistent_data = True
sc.view_settings.view_transform = 'Standard'
world = bpy.data.worlds.new('w'); sc.world = world; world.use_nodes = True
bg = world.node_tree.nodes['Background']; bg.inputs[1].default_value = 0.45
# studio gradient for the chrome to reflect: dark floor, bright ceiling
wn, wl = world.node_tree.nodes, world.node_tree.links
tc = wn.new('ShaderNodeTexCoord'); sep = wn.new('ShaderNodeSeparateXYZ'); ramp = wn.new('ShaderNodeValToRGB')
wl.new(tc.outputs['Generated'], sep.inputs[0]); wl.new(sep.outputs['Z'], ramp.inputs['Fac'])
cr = ramp.color_ramp
cr.elements[0].position = 0.0; cr.elements[0].color = (0.03, 0.03, 0.035, 1)
cr.elements[1].position = 1.0; cr.elements[1].color = (0.45, 0.45, 0.47, 1)
for pos, v in ((0.48, 0.03), (0.56, 0.95), (0.66, 0.18), (0.8, 0.7)):      # dark floor, bright horizon strip, dark band, soft top
    e = cr.elements.new(pos); e.color = (v, v, v * 1.02, 1)
bg.inputs[0].default_value = (0.62, 0.62, 0.64, 1)   # even studio surround: chrome reads as clean silver (gradient left unlinked)

def area(name, loc, energy, size):
    L = bpy.data.lights.new(name, 'AREA'); L.energy = energy; L.size = size
    o = bpy.data.objects.new(name, L); sc.collection.objects.link(o); o.location = loc
    o.rotation_euler = (-mathutils.Vector(loc)).to_track_quat('-Z', 'Y').to_euler(); return o
area('key', (-1.2, -2.2, 2.2), 520, 2.5)
area('fill', (1.8, -0.8, 1.0), 140, 3.0)
area('rim', (-2.0, 1.8, 1.6), 450, 1.5)
bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, FLOOR0)); floor = bpy.context.object
floor.is_shadow_catcher = True
floor.visible_glossy = False          # keep the floor out of the chrome's reflections

cam = bpy.data.cameras.new('cam'); cam.lens = 50
camo = bpy.data.objects.new('cam', cam); sc.collection.objects.link(camo); sc.camera = camo

def camera_at(t):
    az = math.radians(228 + 22 * math.sin(math.pi * t / 4.8))   # orbit out and back: last frame = first frame
    e = smooth((t - T0) / 0.8) * (1 - smooth((t - (HOLD_END + 7 * STEP)) / 0.8))   # pull back as the parts leave, in as they return
    dist = 3.0 + 1.1 * e; el = math.radians(10)
    target = mathutils.Vector((-0.08 - 0.08 * e, 0.0, 0.06 * e))   # aim a little higher while the headrest is up
    pos = target + dist * mathutils.Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
    camo.location = pos; camo.rotation_euler = (target - pos).to_track_quat('-Z', 'Y').to_euler()

def pose(t):
    for n, (i, off) in MOVES.items():
        k = amount(i, t); objs[n].location = (off[0] * k, off[1] * k, off[2] * k)
        objs[n]['glow'] = 4 * k * (1 - k)                 # flares while the part breaks away and while it locks back in
    floor.location.z = FLOOR0 - WHEEL_DROP * amount(7, t)
    camera_at(t)

if MODE == 'parts':
    # each part alone, from two sides, to check that it looks complete on its own
    pose(0.0); floor.hide_render = True; sc.render.film_transparent = True
    sc.render.resolution_x = sc.render.resolution_y = int(os.environ.get('PRES', 300))
    os.makedirs('parts', exist_ok=True)
    order = os.environ.get('ONLY', 'headrest,backrest,frame,lumbar,seat,arm_l,arm_r,mechanism,gas_lift,base,wheel0').split(',')
    for n in order:
        for o in objs.values(): o.hide_render = (o is not objs[n])
        v = np.zeros(len(objs[n].data.vertices) * 3); objs[n].data.vertices.foreach_get('co', v); v = v.reshape(-1, 3)
        c = mathutils.Vector(((v.min(0) + v.max(0)) / 2).tolist()); r = float(np.linalg.norm(v.max(0) - v.min(0))) / 2
        for side, azd in (('a', 228), ('b', 40)):
            az = math.radians(azd); el = math.radians(15); d = 3.1 * r
            pos = c + d * mathutils.Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
            camo.location = pos; camo.rotation_euler = (c - pos).to_track_quat('-Z', 'Y').to_euler()
            sc.render.filepath = f'/home/user/parts/{n}_{side}.png'; bpy.ops.render.render(write_still=True)
        print('part', n, flush=True)
    sys.exit()

import os
frames = [60] if MODE == 'test' else range(int(os.environ.get('F0', 0)), int(os.environ.get('F1', N)) + 1); os.makedirs('frames', exist_ok=True)
for f in frames:
    t0 = time.time(); pose(f / FPS)
    sc.render.filepath = f'/home/user/frames/{f:04d}.png'; sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGBA'
    bpy.ops.render.render(write_still=True)
    print('frame', f, round(time.time() - t0, 1), flush=True)
