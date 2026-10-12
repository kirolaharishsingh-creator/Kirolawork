# Final chair model: the finished Tripo-part chair (finish_tripo_chair.py output) made consistent and clean.
#  - armrests: one master arm (ARM_MASTER, levelled so the pad is flat side to side) mirrored about the seat's centre
#    plane, so both arms are exact mirror images (the two separate fits sat 4 cm apart front-to-back); a solid block
#    closes the open slot between pad and post
#  - materials: Tripo's base-colour maps carry baked light and light-grey patches. They are graded to the real chair's
#    near-black (luminance remapped into BLACK_LO..BLACK_HI, keeps the weave/grain), normal maps kept.
#    Plastic parts roughness 0.5, seat fabric 0.95. Backrest/lumbar mesh: faces inside the moulded rim on the panel's
#    front sheet get a see-through mesh material (alpha MESH_ALPHA), so the spine and backdrop show through like the real chair.
#    Base: chrome where Tripo's metal map says metal (polished silver), black castors elsewhere.
# Usage: python build_final_chair.py in.blend out.blend   (env: ARM_MASTER arm_l, MESH_ALPHA 0.55, BLACK_LO/HI)
import bpy, sys, os, math, numpy as np, mathutils, cv2
src, dst = sys.argv[-2:]
bpy.ops.wm.open_mainfile(filepath=src)
O = bpy.data.objects
E = lambda k, d: float(os.environ.get(k, d))

def verts(o):
    v = np.zeros(len(o.data.vertices) * 3); o.data.vertices.foreach_get('co', v); return v.reshape(-1, 3)

# ---------------------------------------------------------------- armrests
MASTER = os.environ.get('ARM_MASTER', 'arm_l'); OTHER = 'arm_r' if MASTER == 'arm_l' else 'arm_l'
s = verts(O['seat']); XC = float((s[:, 0].min() + s[:, 0].max()) / 2)
am = O[MASTER]; v = verts(am); top = v[v[:, 2] > v[:, 2].max() - 0.03]; c = top.mean(0)
n = np.linalg.svd(top - c, full_matrices=False)[2][2]; n *= np.sign(n[2])
tilt = math.atan2(n[0], n[2])                              # pad roll about the chair's front-back axis (y)
bot = v[v[:, 2] < v[:, 2].min() + 0.03].mean(0)
Rl = mathutils.Matrix.Translation(bot) @ mathutils.Matrix.Rotation(-tilt, 4, 'Y') @ mathutils.Matrix.Translation(-bot)
am.data.transform(Rl); am.data.update()
# the pad sits on the post with an open slot on the inner side: fill it with a solid block the post's size
import bmesh
v = verts(am); zt = v[:, 2].max(); sl = v[(v[:, 2] > zt - 0.065) & (v[:, 2] < zt - 0.05)]
cx_, cy_ = np.median(sl[:, 0]), np.median(sl[:, 1]); core = sl[(abs(sl[:, 0] - cx_) < 0.03) & (abs(sl[:, 1] - cy_) < 0.04)]
lo3, hi3 = core.min(0), core.max(0); shrink = 0.002
bm = bmesh.new(); bm.from_mesh(am.data); r_ = bmesh.ops.create_cube(bm, size=1.0)
bmesh.ops.scale(bm, verts=r_['verts'], vec=(hi3[0] - lo3[0] - 2 * shrink, hi3[1] - lo3[1] - 2 * shrink, 0.05))
bmesh.ops.translate(bm, verts=r_['verts'], vec=((lo3[0] + hi3[0]) / 2, (lo3[1] + hi3[1]) / 2, zt - 0.04))
bm.to_mesh(am.data); bm.free(); am.data.update()
print('ARM PAD BLOCK', (hi3 - lo3).round(3), 'z', round(zt - 0.065, 3), '-', round(zt - 0.015, 3))
old = O[OTHER]; omats = list(old.data.materials); bpy.data.objects.remove(old, do_unlink=True)
me = am.data.copy(); me.name = OTHER
me.transform(mathutils.Matrix.Translation((XC, 0, 0)) @ mathutils.Matrix.Scale(-1, 4, (1, 0, 0)) @ mathutils.Matrix.Translation((-XC, 0, 0)))
me.flip_normals(); me.update()
oa = O.new(OTHER, me); bpy.context.scene.collection.objects.link(oa)
print('ARMS master', MASTER, 'pad roll fixed', round(math.degrees(tilt), 1), 'deg; mirror plane x =', round(XC, 4))

# ---------------------------------------------------------------- materials
LO, HI = E('BLACK_LO', 0.006), E('BLACK_HI', 0.05)

def panel_faces(o, rim, depth):
    # faces of the panel's mesh area: inside the outline (2-D distance to the edge > rim, in the panel's best-fit plane)
    # and on the front sheet (within `depth` of the front-most surface in that cell), so brackets behind stay solid
    me = o.data; n = len(me.polygons)
    c = np.zeros(n * 3); me.polygons.foreach_get('center', c); c = c.reshape(-1, 3)
    c0 = c.mean(0); U, S_, Vt = np.linalg.svd(c[::7] - c0, full_matrices=False); N = Vt[2] * (-1 if Vt[2][1] > 0 else 1)  # front = -y
    uv = np.c_[(c - c0) @ Vt[0], (c - c0) @ Vt[1]]; d = (c - c0) @ N
    g = 0.002; lo_ = uv.min(0); ij = ((uv - lo_) / g).astype(int) + 4; W_, H_ = ij.max(0) + 6
    occ = np.zeros((H_, W_), np.uint8); occ[ij[:, 1], ij[:, 0]] = 1
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    solid = (cv2.floodFill(occ.copy(), None, (0, 0), 2)[1] != 2).astype(np.uint8)       # outline with holes filled
    dist = cv2.distanceTransform(solid, cv2.DIST_L2, 5) * g
    front = np.full((H_, W_), -1e9); np.maximum.at(front, (ij[:, 1], ij[:, 0]), d)
    front = cv2.dilate(front.astype(np.float32), np.ones((5, 5), np.uint8))
    sel = (dist[ij[:, 1], ij[:, 0]] > rim) & (d > front[ij[:, 1], ij[:, 0]] - depth)
    print('PANEL', o.name, 'faces', int(sel.sum()), 'of', n); return sel

PANELS = {'backrest': (E('RIM_BACK', 0.03), 0.012), 'lumbar': (E('RIM_LUMB', 0.02), 0.01)}   # (rim width, front-sheet depth)
FABRIC = {'tp_seat'}
for m in bpy.data.materials:
    if not m.name.startswith('tp_') or not m.node_tree: continue
    nt, L = m.node_tree, m.node_tree.links; pb = [x for x in nt.nodes if x.type == 'BSDF_PRINCIPLED'][0]
    tex = [x for x in nt.nodes if x.type == 'TEX_IMAGE' and x.image.colorspace_settings.name == 'sRGB'][0]
    rm = [x for x in nt.nodes if x.type == 'TEX_IMAGE' and '_rm' in x.image.name]
    for x in [x for x in nt.nodes if x.type in ('MIX', 'MAP_RANGE', 'MATH')]: nt.nodes.remove(x)   # finish_tripo_chair's grading
    for sock in ('Base Color', 'Roughness', 'Metallic', 'Alpha'):
        for l in list(pb.inputs[sock].links): L.remove(l)
    bw = nt.nodes.new('ShaderNodeRGBToBW'); L.new(tex.outputs['Color'], bw.inputs[0])
    gr = nt.nodes.new('ShaderNodeMapRange'); gr.clamp = True; L.new(bw.outputs[0], gr.inputs['Value'])
    gr.inputs['From Min'].default_value, gr.inputs['From Max'].default_value = 0.02, 0.45
    gr.inputs['To Min'].default_value, gr.inputs['To Max'].default_value = LO, HI
    if m.name == 'tp_base' and rm:
        sep = nt.nodes.new('ShaderNodeSeparateColor'); L.new(rm[0].outputs['Color'], sep.inputs[0])
        mk = nt.nodes.new('ShaderNodeMapRange'); mk.clamp = True; L.new(sep.outputs['Blue'], mk.inputs['Value'])
        mk.inputs['From Min'].default_value, mk.inputs['From Max'].default_value = 0.35, 0.6
        col = nt.nodes.new('ShaderNodeMix'); col.data_type = 'RGBA'; L.new(mk.outputs['Result'], col.inputs[0])
        L.new(gr.outputs['Result'], col.inputs[6]); col.inputs[7].default_value = (0.92, 0.92, 0.93, 1)
        L.new(col.outputs[2], pb.inputs['Base Color']); L.new(mk.outputs['Result'], pb.inputs['Metallic'])
        ro = nt.nodes.new('ShaderNodeMix'); ro.data_type = 'FLOAT'; L.new(mk.outputs['Result'], ro.inputs[0])
        ro.inputs[2].default_value, ro.inputs[3].default_value = 0.5, 0.06; L.new(ro.outputs[0], pb.inputs['Roughness'])
        print('MAT', m.name, 'chrome + black castors'); continue
    if m.name in FABRIC: gr.inputs['To Max'].default_value = E('SEAT_HI', 0.028)
    L.new(gr.outputs['Result'], pb.inputs['Base Color']); pb.inputs['Metallic'].default_value = 0.0
    pb.inputs['Roughness'].default_value = 0.95 if m.name in FABRIC else 0.5
    if m.name in ('tp_backrest', 'tp_lumbar'):
        o = O[m.name[3:]]; rim, depth = PANELS[o.name]; sel = panel_faces(o, rim, depth)
        mm = m.copy(); mm.name = m.name + '_mesh'; pm = [x for x in mm.node_tree.nodes if x.type == 'BSDF_PRINCIPLED'][0]
        pm.inputs['Alpha'].default_value = E('MESH_ALPHA', 0.55); pm.inputs['Roughness'].default_value = 0.85
        o.data.materials.append(mm); k = len(o.data.materials) - 1
        mi = np.zeros(len(o.data.polygons), np.int32); o.data.polygons.foreach_get('material_index', mi)
        mi[sel] = k; o.data.polygons.foreach_set('material_index', mi); o.data.update()
    print('MAT', m.name, 'graded', 'fabric' if m.name in FABRIC else 'plastic')

for n_ in ('frame_plastic',):
    m = bpy.data.materials.get(n_)
    if m:
        pb = [x for x in m.node_tree.nodes if x.type == 'BSDF_PRINCIPLED'][0]
        pb.inputs['Base Color'].default_value = (0.012, 0.012, 0.013, 1); pb.inputs['Roughness'].default_value = 0.5

bpy.ops.wm.save_as_mainfile(filepath=dst, compress=True); print('saved', dst)
