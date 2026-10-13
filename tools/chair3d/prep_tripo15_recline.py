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
LUMB_MODE = os.environ.get('LUMB_MODE', 'tilt')     # tilt: the pad rocks on the pivot rod at the arm's front end (arm stays
                                                     # on the spine); slide: pad + arm slide together on the rod
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

# lumbar depth slide: a fixed rod from the spine into the pivot arm (part 10), hidden inside the arm at rest, shows as
# the pad + arm slide forward (frame__lumbrod follows the frame)
a10 = V(P['tripo_part_10']); yz = a10[:, 1:]; cyz = yz.mean(0); d2 = np.linalg.svd(yz - cyz, full_matrices=False)[2][0]
if d2[0] < 0: d2 = -d2                                         # side-view direction of the arm, +y = towards the spine
ax10 = np.array([0.0, d2[0], d2[1]]); t10 = (yz - cyz) @ d2; end = a10[t10 > t10.max() - 0.012]
rear = np.array([np.median(end[:, 0]), *(cyz + d2 * t10.max())])
r10 = float(min(np.ptp(end[:, 0]), np.ptp(end[:, 2])) / 2 * 0.6)
ROD = float(os.environ.get('ROD', 0.06 if LUMB_MODE == 'slide' else 0))
if ROD:
    rb = bmesh.new(); rr = bmesh.ops.create_cone(rb, cap_ends=True, segments=24, radius1=r10, radius2=r10, depth=ROD)
    q = mathutils.Vector((0, 0, 1)).rotation_difference(mathutils.Vector(ax10.tolist()))
    bmesh.ops.rotate(rb, verts=rr['verts'], cent=(0, 0, 0), matrix=q.to_matrix())
    bmesh.ops.translate(rb, verts=rr['verts'], vec=mathutils.Vector((rear - ax10 * (ROD / 2 - 0.012)).tolist()))
    rme = bpy.data.meshes.new('frame__lumbrod'); rb.to_mesh(rme); rb.free()
    rm_ = bpy.data.materials.new('tp_lumbrod'); rm_.use_nodes = True; rp_ = rm_.node_tree.nodes['Principled BSDF']
    rp_.inputs['Base Color'].default_value = (0.022, 0.022, 0.024, 1); rp_.inputs['Roughness'].default_value = 0.45; rme.materials.append(rm_)
    for pl in rme.polygons: pl.use_smooth = True
    P['frame__lumbrod'] = bpy.data.objects.new('frame__lumbrod', rme); bpy.context.scene.collection.objects.link(P['frame__lumbrod'])
front = np.array([np.median(a10[t10 < t10.min() + 0.012][:, 0]), *(cyz + d2 * t10.min())])
print('LUMB_PIVOT=%.4f,%.4f,%.4f' % tuple(front), flush=True)
print('LUMB ROD r', round(r10, 4), 'axis', ax10.round(3), 'rear', rear.round(3), flush=True)

def join(names, new):
    obs = [P[n] for n in names]; bpy.ops.object.select_all(action='DESELECT')
    for ob in obs: ob.select_set(True)
    bpy.context.view_layer.objects.active = obs[0]
    if len(obs) > 1: bpy.ops.object.join()
    obs[0].name = new; obs[0].data.name = new; return obs[0]
NAMES = {'backrest': ['tripo_part_0'], 'frame': ['tripo_part_6', 'tripo_part_3'] + (['tripo_part_10'] if LUMB_MODE == 'tilt' else []),
         'headrest': ['tripo_part_9'], 'lumbar': ['tripo_part_2'] + (['tripo_part_10'] if LUMB_MODE == 'slide' else []),
 'seat': ['tripo_part_5'], 'mechanism': ['tripo_part_4'], 'base': ['tripo_part_1']}
arms = sorted(['tripo_part_7', 'tripo_part_8'], key=lambda n: C(P[n])[0]); NAMES['arm_l'] = [arms[0]]; NAMES['arm_r'] = [arms[1]]
wheels = sorted(['tripo_part_11', 'tripo_part_12', 'tripo_part_13', 'tripo_part_14'], key=lambda n: math.atan2(C(P[n])[1], C(P[n])[0]))
for i, n in enumerate(wheels): NAMES['wheel_%d' % i] = [n]
OB = {k: join(v, k) for k, v in NAMES.items()}

# a loose plate under the lumbar loop's lower tip (an extra element, not on the real chair): remove the loose mesh pieces that
# lie entirely in the box under the tip (LUMB_DROP='y0,z1': y > y0 and z < z1), the loop itself stays untouched
DROP = os.environ.get('LUMB_DROP', '')       # off: the 'extra element' was the 2D gap fill, the model's lumbar is fine
if DROP:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    y0, z1 = map(float, DROP.split(',')); me_ = OB['lumbar'].data
    vv = np.zeros(len(me_.vertices) * 3); me_.vertices.foreach_get('co', vv); vv = vv.reshape(-1, 3)
    ee = np.zeros(len(me_.edges) * 2, np.int64); me_.edges.foreach_get('vertices', ee); ee = ee.reshape(-1, 2)
    nl, lab = connected_components(coo_matrix((np.ones(len(ee)), (ee[:, 0], ee[:, 1])), shape=(len(vv), len(vv))), directed=False)
    bad = [i for i in range(nl) if (lambda q: q[:, 1].min() > y0 and q[:, 2].max() < z1 and q[:, 2].min() < 0.4)(vv[lab == i])]
    kill = np.isin(lab, bad); lb = bmesh.new(); lb.from_mesh(me_); lb.verts.ensure_lookup_table()
    bmesh.ops.delete(lb, geom=[lb.verts[i] for i in np.where(kill)[0]], context='VERTS'); lb.to_mesh(me_); lb.free(); me_.update()
    print('LUMBAR dropped pieces', bad, 'verts', int(kill.sum()), flush=True)

# armrest 4D side slide: each pad splits off its post (faces above the height where the arm's cross-section widens into
# the pad) as arm_x__pad, which follows its arm in anim3 and can slide sideways (ARM_SIDE)
if os.environ.get('ARM_SPLIT', '1') == '1':
    for side in ('arm_l', 'arm_r'):
        ob = OB[side]; va = V(ob); zt = None
        for z in np.arange(0.45, 0.6, 0.0025):
            sl_ = va[abs(va[:, 2] - z) < 0.002]
            if len(sl_) and np.ptp(sl_[:, 1]) > 0.10: zt = z - 0.006; break
        mp = ob.data.copy(); mp.name = side + '__pad'; op = bpy.data.objects.new(side + '__pad', mp); bpy.context.scene.collection.objects.link(op)
        for target, keep_top in ((ob, False), (op, True)):
            bm_ = bmesh.new(); bm_.from_mesh(target.data); bm_.faces.ensure_lookup_table()
            bmesh.ops.delete(bm_, geom=[f for f in bm_.faces if (f.calc_center_median().z > zt) != keep_top], context='FACES')
            bmesh.ops.delete(bm_, geom=[x for x in bm_.verts if not x.link_faces], context='VERTS')
            bd_ = [e for e in bm_.edges if e.is_boundary and all(abs(v.co.z - zt) < 0.01 for v in e.verts)]
            if bd_: bmesh.ops.holes_fill(bm_, edges=bd_, sides=0)
            bm_.to_mesh(target.data); bm_.free(); target.data.update()
        print('ARM PAD split', side, 'at z', round(zt, 4), flush=True)

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
