# Recline-only model from the client's single-mesh chair (user_chair.glb, best side profile): its 552 loose pieces are
# sorted into the three recline groups by position and saved as anim3.py parts:
#   backrest = back unit (backrest, spine incl. its lower J to the mechanism, lumbar, headrest)  -> turns back
#   seat     = seat cushion, seat plates and both armrests                                         -> synchro tilt
#   base     = mechanism, gas lift, five-star base, castors                                         -> fixed
# anim3's other part names become tiny hidden stand-ins (gas_lift one sits on the column axis so the turntable hub is right).
# Material: one Tripo PBR texture, graded like build_final_chair.py: chrome where the metal map says metal, true black
# elsewhere (named tp_base so anim3's CHROME option polishes it). Prints the recline pivots (RPIVOT / SPIVOT for anim3).
# Usage: python prep_user_recline.py user_chair.glb out.blend
import bpy, sys, numpy as np, mathutils
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
src, dst = sys.argv[-2:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
o = [o for o in bpy.data.objects if o.type == 'MESH'][0]; me = o.data
for x in [x for x in bpy.data.objects if x is not o]: bpy.data.objects.remove(x, do_unlink=True)
o.parent = None; me.transform(o.matrix_world); o.matrix_world.identity()
v = np.zeros(len(me.vertices) * 3); me.vertices.foreach_get('co', v); v = v.reshape(-1, 3)
e = np.zeros(len(me.edges) * 2, np.int64); me.edges.foreach_get('vertices', e); e = e.reshape(-1, 2)
n, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(v), len(v))), directed=False)
cls = np.zeros(n, int)        # 0 base, 1 seat unit, 2 back unit   (model: x side, y back, z up)
for i in range(n):
    s = v[lab == i]; lo, hi = s.min(0), s.max(0); c = (lo + hi) / 2
    if hi[2] < 0.30 or (c[2] < 0.31 and abs(c[0]) < 0.12 and c[1] < 0.15): cls[i] = 0
    elif abs(c[0]) > 0.185 and 0.29 < c[2] < 0.66: cls[i] = 1                       # armrests
    elif lo[1] > 0.10 and hi[2] < 0.37: cls[i] = 2                                   # spine foot + its mount behind the seat
    elif c[1] < 0.06 and 0.45 < c[2] < 0.66: cls[i] = 1                              # armrest pad bits in front of the back
    elif (hi[2] <= 0.412 and lo[1] < 0.13) or (c[2] < 0.36 and c[1] < 0.14): cls[i] = 1
    else: cls[i] = 2
vc = cls[lab]
fv = np.array([p.vertices[0] for p in me.polygons]); fcls = vc[fv]

import bmesh
def piece(name, k):
    m2 = me.copy(); m2.name = name; ob = bpy.data.objects.new(name, m2); bpy.context.scene.collection.objects.link(ob)
    bm = bmesh.new(); bm.from_mesh(m2); bm.faces.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[f for f, c in zip(bm.faces, fcls) if c != k], context='FACES')
    bmesh.ops.delete(bm, geom=[x for x in bm.verts if not x.link_faces], context='VERTS')
    bm.to_mesh(m2); bm.free(); return ob
base, seat, back = piece('base', 0), piece('seat', 1), piece('backrest', 2)
bpy.data.objects.remove(o, do_unlink=True)

# pivots: back unit turns at the spine foot under the seat's rear edge; the seat tilts about the mechanism's front edge
RP = np.array([0.0, 0.14, 0.32])          # where the spine foot meets the seat's rear underside (turning here keeps them joined)
vm = v[(vc == 0) & (v[:, 2] > 0.25) & (abs(v[:, 0]) < 0.12)]; SP = np.array([0.0, vm[:, 1].min(), vm[:, 2].max()])
col = v[(vc == 0) & (v[:, 2] > 0.15) & (v[:, 2] < 0.24)]; hub = np.median(col, 0)      # median: the tilt lever sticks out
print('RPIVOT=%.4f,%.4f,%.4f SPIVOT=%.4f,%.4f,%.4f' % (0.0, RP[1], RP[2], 0.0, SP[1], SP[2]), 'hub', hub.round(3), flush=True)

def stub(name, at, hide=True):
    m = bpy.data.meshes.new(name); m.from_pydata([at, (at[0] + 1e-4, at[1], at[2]), (at[0], at[1] + 1e-4, at[2])], [], [(0, 1, 2)])
    ob = bpy.data.objects.new(name, m); bpy.context.scene.collection.objects.link(ob); ob.hide_render = hide; return ob
stub('gas_lift', (hub[0], hub[1], 0.2))
for nme in ('headrest', 'frame', 'lumbar', 'arm_l', 'arm_r', 'mechanism'): stub(nme, (0, 0, 0.4))
for i, a in enumerate(np.linspace(0, 2 * np.pi, 6)[:5]): stub('wheel_%d' % i, (0.2 * np.cos(a), 0.2 * np.sin(a), 0.03))

# material: chrome where metal, graded true black elsewhere
m = me.materials[0]; m.name = 'tp_base'; nt, L = m.node_tree, m.node_tree.links; pb = [x for x in nt.nodes if x.type == 'BSDF_PRINCIPLED'][0]
tex = pb.inputs['Base Color'].links[0].from_node
mr = pb.inputs['Metallic'].links[0].from_socket if pb.inputs['Metallic'].links else None
for sck in ('Base Color', 'Roughness', 'Metallic'):
    for l in list(pb.inputs[sck].links): L.remove(l)
bw = nt.nodes.new('ShaderNodeRGBToBW'); L.new(tex.outputs['Color'], bw.inputs[0])
gr = nt.nodes.new('ShaderNodeMapRange'); gr.clamp = True; L.new(bw.outputs[0], gr.inputs['Value'])
gr.inputs['From Min'].default_value, gr.inputs['From Max'].default_value = 0.02, 0.5
gr.inputs['To Min'].default_value, gr.inputs['To Max'].default_value = 0.006, 0.05
mk = nt.nodes.new('ShaderNodeMapRange'); mk.clamp = True; mk.inputs['From Min'].default_value, mk.inputs['From Max'].default_value = 0.35, 0.6
if mr: L.new(mr, mk.inputs['Value'])
else: mk.inputs['Value'].default_value = 0
col_ = nt.nodes.new('ShaderNodeMix'); col_.data_type = 'RGBA'; L.new(mk.outputs['Result'], col_.inputs[0])
L.new(gr.outputs['Result'], col_.inputs[6]); col_.inputs[7].default_value = (0.92, 0.92, 0.93, 1)
L.new(col_.outputs[2], pb.inputs['Base Color']); L.new(mk.outputs['Result'], pb.inputs['Metallic'])
ro = nt.nodes.new('ShaderNodeMix'); ro.data_type = 'FLOAT'; L.new(mk.outputs['Result'], ro.inputs[0])
ro.inputs[2].default_value, ro.inputs[3].default_value = 0.55, 0.06; L.new(ro.outputs[0], pb.inputs['Roughness'])
print('MAT metal map', 'linked' if mr else 'none', flush=True)
bpy.ops.wm.save_as_mainfile(filepath=dst, compress=True); print('saved', dst, flush=True)
