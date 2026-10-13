# Recline model from the client's 15-part Tripo chair (re-export of tripo_chair_clean_front_pbr): best side profile and
# already split into parts, so the back unit tilts as whole parts (no cut through a single mesh).
#  - straightened: turned about z so the armrests sit on the x axis (front = -y, back = +y), column on the origin
#  - parts renamed for anim3.py: backrest, frame (spine + its foot), headrest, lumbar (+ knob), seat, arm_l/arm_r,
#    mechanism, base, wheel_0..4 (the base mesh carries one castor -> wheel_4 is a hidden stand-in), gas_lift stand-in
#  - materials graded like build_final_chair.py: true black (patchy light texels gone), chrome where the metal map says
#    the base's legs and hub (no metal map in this export; its castor faces black; named tp_base for anim3's CHROME option)
# Prints RPIVOT / SPIVOT for anim3's RECLINE mode.
# Usage: python prep_tripo15_recline.py model.glb out.blend
import bpy, bmesh, sys, os, math, numpy as np, mathutils
src, dst = sys.argv[-2:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
P = {}
for o in [o for o in bpy.data.objects if o.type == 'MESH']:
    o.parent = None; o.data.transform(o.matrix_world); o.matrix_world.identity(); P[o.name] = o
for o in [o for o in bpy.data.objects if o.type != 'MESH']: bpy.data.objects.remove(o, do_unlink=True)
def V(o): a = np.zeros(len(o.data.vertices) * 3); o.data.vertices.foreach_get('co', a); return a.reshape(-1, 3)
def C(o): a = V(o); return (a.min(0) + a.max(0)) / 2

# the backrest rim carries raised lettering (a logo the generator modelled as relief) that catches the light once the back
# reclines: smooth fine relief off the backrest (SMOOTH iterations of Laplacian-style smoothing; 0 = off)
SM = int(os.environ.get('SMOOTH', 0))     # off: the backrest is many loose pieces and smoothing opens cracks
if SM:
    for n in ('tripo_part_0',):
        o_ = P[n]; m_ = o_.modifiers.new('sm', 'SMOOTH'); m_.factor = 0.7; m_.iterations = SM
        bpy.context.view_layer.objects.active = o_; bpy.ops.object.modifier_apply(modifier='sm')
    print('SMOOTHED backrest', SM, flush=True)

# chrome band over the flat cut where the upper spine (part 6) meets its lower curve (part 3), like the real chair's ring:
# both parts' faces within BAND of the cut plane, pushed out 1 mm along their normals, polished chrome (frame__collar
# follows the frame in anim3)
def V0(o): a = np.zeros(len(o.data.vertices) * 3); o.data.vertices.foreach_get('co', a); return a.reshape(-1, 3)
from scipy.spatial import cKDTree
a6, a3 = V0(P['tripo_part_6']), V0(P['tripo_part_3']); seam = a6[cKDTree(a3).query(a6)[0] < 0.004]
ZC = float(np.median(seam[:, 2])); BAND = float(os.environ.get('BAND', 0.006)); lo_, hi_ = seam.min(0) - 0.02, seam.max(0) + 0.02
cbm = bmesh.new()
for n in ('tripo_part_6', 'tripo_part_3'):
    t = bmesh.new(); t.from_mesh(P[n].data); t.faces.ensure_lookup_table()
    keep = lambda c: abs(c.z - ZC) < BAND and lo_[0] < c.x < hi_[0] and lo_[1] < c.y < hi_[1]
    bmesh.ops.delete(t, geom=[f for f in t.faces if not keep(f.calc_center_median())], context='FACES')
    bmesh.ops.delete(t, geom=[x for x in t.verts if not x.link_faces], context='VERTS')
    t.normal_update()
    for x in t.verts: x.co += x.normal * 0.001
    me_ = bpy.data.meshes.new('tmp'); t.to_mesh(me_); t.free(); cbm.from_mesh(me_); bpy.data.meshes.remove(me_)
cme = bpy.data.meshes.new('frame__collar'); cbm.to_mesh(cme); cbm.free()
cm = bpy.data.materials.new('tp_chrome'); cm.use_nodes = True; pbc = cm.node_tree.nodes['Principled BSDF']
pbc.inputs['Base Color'].default_value = (0.92, 0.92, 0.93, 1); pbc.inputs['Metallic'].default_value = 1.0; pbc.inputs['Roughness'].default_value = 0.08
cme.materials.append(cm); P['frame__collar'] = bpy.data.objects.new('frame__collar', cme); bpy.context.scene.collection.objects.link(P['frame__collar'])
for pl in cme.polygons: pl.use_smooth = True
print('COLLAR at z', round(ZC, 4), 'faces', len(cme.polygons), flush=True)

# straighten: armrest centres on the x axis, spine behind (+y)
a7, a8 = C(P['tripo_part_7']), C(P['tripo_part_8']); d = a8 - a7
yaw = -math.atan2(d[1], d[0]) + math.radians(float(os.environ.get('YAW_ADJ', 4))); Rz = mathutils.Matrix.Rotation(yaw, 4, 'Z')
for o in P.values(): o.data.transform(Rz)
if C(P['tripo_part_6'])[1] < C(P['tripo_part_5'])[1]:            # spine must be behind the seat
    for o in P.values(): o.data.transform(mathutils.Matrix.Rotation(math.pi, 4, 'Z'))
b = V(P['tripo_part_1']); col = b[(b[:, 2] > 0.15)]; hub = np.median(col, 0)
for o in P.values(): o.data.transform(mathutils.Matrix.Translation((-hub[0], -hub[1], 0)))
print('YAW deg', round(math.degrees(yaw), 1), 'hub', hub.round(3), flush=True)

def join(names, new):
    obs = [P[n] for n in names]; bpy.ops.object.select_all(action='DESELECT')
    for ob in obs: ob.select_set(True)
    bpy.context.view_layer.objects.active = obs[0]
    if len(obs) > 1: bpy.ops.object.join()
    obs[0].name = new; obs[0].data.name = new; return obs[0]
NAMES = {'backrest': ['tripo_part_0'], 'frame': ['tripo_part_6', 'tripo_part_3'], 'headrest': ['tripo_part_9'],
         'lumbar': ['tripo_part_2', 'tripo_part_10'], 'seat': ['tripo_part_5'], 'mechanism': ['tripo_part_4'], 'base': ['tripo_part_1']}
arms = sorted(['tripo_part_7', 'tripo_part_8'], key=lambda n: C(P[n])[0]); NAMES['arm_l'] = [arms[0]]; NAMES['arm_r'] = [arms[1]]
wheels = sorted(['tripo_part_11', 'tripo_part_12', 'tripo_part_13', 'tripo_part_14'], key=lambda n: math.atan2(C(P[n])[1], C(P[n])[0]))
for i, n in enumerate(wheels): NAMES['wheel_%d' % i] = [n]
OB = {k: join(v, k) for k, v in NAMES.items()}

# the lumbar loop's lower tip ends in a square-cut stub that pokes out under the loop: cut the tip square to its own
# direction and cap it flat
TRIM = os.environ.get('LUMB_TRIM', '0.230,0.412')        # cut point (y, z) on the tip; '' = off
if TRIM:
    ty, tz = map(float, TRIM.split(',')); ax = mathutils.Vector((0, 0.61, -0.79)).normalized()   # tip direction (down/back)
    lb = bmesh.new(); lb.from_mesh(OB['lumbar'].data); lb.faces.ensure_lookup_table()
    beyond = lambda c: (c.y - ty) * ax.y + (c.z - tz) * ax.z > 0 and c.y > ty - 0.03 and c.z < tz + 0.03
    cut = [f for f in lb.faces if beyond(f.calc_center_median())]
    bmesh.ops.delete(lb, geom=cut, context='FACES'); bmesh.ops.delete(lb, geom=[x for x in lb.verts if not x.link_faces], context='VERTS')
    bd = [e for e in lb.edges if e.is_boundary and all(abs((v.co.y - ty) * ax.y + (v.co.z - tz) * ax.z) < 0.006 and v.co.y > ty - 0.03 for v in e.verts)]
    if bd: bmesh.ops.holes_fill(lb, edges=bd, sides=0)
    lb.to_mesh(OB['lumbar'].data); lb.free(); OB['lumbar'].data.update(); print('LUMBAR tip cut faces', len(cut), 'capped edges', len(bd), flush=True)

def stub(name, at):
    m = bpy.data.meshes.new(name); m.from_pydata([at, (at[0] + 1e-4, at[1], at[2]), (at[0], at[1] + 1e-4, at[2])], [], [(0, 1, 2)])
    ob = bpy.data.objects.new(name, m); bpy.context.scene.collection.objects.link(ob); ob.hide_render = True; return ob
stub('gas_lift', (0, 0, 0.2))
bb = V(OB['base']); far = bb[(bb[:, 2] < 0.07)]; ang = np.arctan2(far[:, 1], far[:, 0]); r = np.hypot(far[:, 0], far[:, 1])
stub('wheel_4', tuple(far[np.argmax(r)]))

# materials
LO, HI = 0.006, 0.05
import os; FLAT = float(os.environ.get('FLAT', 0.022))      # >0: one flat black (texture baked blotches gone), detail from geometry/normals
for k, ob in OB.items():
    for m in ob.data.materials:
        m.name = 'tp_base' if k == 'base' else 'tp_' + k
        nt, L = m.node_tree, m.node_tree.links; pb = [x for x in nt.nodes if x.type == 'BSDF_PRINCIPLED'][0]
        tex = pb.inputs['Base Color'].links[0].from_node if pb.inputs['Base Color'].links else None
        met = pb.inputs['Metallic'].links[0].from_socket if pb.inputs['Metallic'].links else None
        if os.environ.get('NONORMAL', '1') == '1':      # Tripo's normal maps carry the dark blotches (pads, seat, lumbar, knob)
            for l in list(pb.inputs['Normal'].links): L.remove(l)
        for s in ('Base Color', 'Roughness', 'Metallic'):
            for l in list(pb.inputs[s].links): L.remove(l)
        gr = nt.nodes.new('ShaderNodeMapRange'); gr.clamp = True
        gr.inputs['From Min'].default_value, gr.inputs['From Max'].default_value = 0.02, 0.5
        gr.inputs['To Min'].default_value, gr.inputs['To Max'].default_value = (FLAT, FLAT) if FLAT else (LO, HI * (0.6 if k == 'seat' else 1))
        if tex: bw = nt.nodes.new('ShaderNodeRGBToBW'); L.new(tex.outputs['Color'], bw.inputs[0]); L.new(bw.outputs[0], gr.inputs['Value'])
        else: gr.inputs['Value'].default_value = 0.1
        if k == 'base':                  # no metal map in this export: legs + hub polished chrome, its castor black (below)
            pb.inputs['Base Color'].default_value = (0.92, 0.92, 0.93, 1); pb.inputs['Metallic'].default_value = 1.0
            pb.inputs['Roughness'].default_value = 0.06
        else:
            L.new(gr.outputs['Result'], pb.inputs['Base Color']); pb.inputs['Metallic'].default_value = 0.0
            pb.inputs['Roughness'].default_value = 0.9 if k == 'seat' else 0.55

# the base mesh carries one castor: its faces (low and far out) get a black copy of a castor material
bo = OB['base']; cm_ = OB['wheel_0'].data.materials[0]; bo.data.materials.append(cm_)
fc = np.zeros(len(bo.data.polygons) * 3); bo.data.polygons.foreach_get('center', fc); fc = fc.reshape(-1, 3)
mi = np.zeros(len(fc), np.int32); mi[((fc[:, 2] < 0.065) & (np.hypot(fc[:, 0], fc[:, 1]) > 0.2)) | ((fc[:, 2] > 0.13) & (np.hypot(fc[:, 0], fc[:, 1]) < 0.045))] = len(bo.data.materials) - 1   # + the gas-lift column
bo.data.polygons.foreach_set('material_index', mi); bo.data.update(); print('BASE castor faces', int((mi > 0).sum()), flush=True)

# pivots: back unit turns at the spine foot under the seat's rear edge; seat tilts about the mechanism's front top
s = V(OB['seat']); m = V(OB['mechanism'])
RP = (0.0, float(s[:, 1].max() - 0.03), float(s[:, 2].min() + 0.01)); SP = (0.0, float(m[:, 1].min()), float(m[:, 2].max()))
print('RPIVOT=%.4f,%.4f,%.4f SPIVOT=%.4f,%.4f,%.4f' % (RP + SP), flush=True)
for k, ob in OB.items(): print('PART', k, C(ob).round(3), flush=True)
bpy.ops.wm.save_as_mainfile(filepath=dst, compress=True); print('saved', dst, flush=True)
