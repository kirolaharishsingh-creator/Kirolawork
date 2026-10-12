# Rig the final chair (build_final_chair.py output): split the moving pieces and add Blender controls for every movement.
#  Split (followers keep the family prefix 'family__piece', anim3.py parents them to the family part):
#   - arm_l / arm_r -> arm_x (bracket + lower tube), arm_x__upper (sleeve), arm_x__pad, arm_x__tube (inner tube, hidden
#     inside the sleeve at rest, shows when the arm is raised); cut faces are capped
#   - headrest lock (frame_plastic faces of the headrest) -> frame__lock: stays on the spine when the headrest slides
#  Controls: empty CHAIR_RIG carries the movement properties; drivers move control empties that carry the parts:
#   recline_deg   0..45   seat-to-back angle increase; backrest group turns about the recline pivot, seat tilts
#                         along (synchro ratio SYNC) about the seat pivot, arms ride on the seat
#   seat_slide    0..0.05 m  seat cushion forward
#   lumbar_depth  0..0.04 m  lumbar pad forward (in / out)
#   arm_height    0..0.07 m  sleeve + pad up along the post axis
#   arm_depth  -0.03..0.03 m pad forward (+) / back
#   arm_width  -0.02..0.025 m pad sideways, outward (+) / inward
#   arm_rotate_deg -20..20   pad turns about its vertical axis, front inward (+)
#   headrest_height 0..0.06 m headrest up
#  Each moving piece also stores 'move_pivot' / 'move_axis' (model coordinates) for scripts that animate without drivers.
# Usage: python rig_final_chair.py in.blend out.blend
import bpy, bmesh, sys, math, json, numpy as np, mathutils
src, dst = sys.argv[-2:]
bpy.ops.wm.open_mainfile(filepath=src)
O = bpy.data.objects; sc = bpy.context.scene; V = mathutils.Vector
SYNC = 0.18; RPIVOT = V((0, 0.13, 0.40)); SPIVOT = V((0, -0.13, 0.32))
Z_TUBE, Z_PAD = 0.415, 0.505                 # arm cuts: lower tube | sleeve | pad (from the arm's height slices)

def fcentres(me):
    c = np.zeros(len(me.polygons) * 3); me.polygons.foreach_get('center', c); return c.reshape(-1, 3)

def split(o, keep_new, name):
    # faces where keep_new go to a new object `name`; both pieces get their cut capped
    me2 = o.data.copy(); me2.name = name; o2 = O.new(name, me2); sc.collection.objects.link(o2)
    for ob, inv in ((o, True), (o2, False)):
        bm = bmesh.new(); bm.from_mesh(ob.data); bm.faces.ensure_lookup_table()
        sel = keep_new(np.array([f.calc_center_median()[:] for f in bm.faces]))
        bmesh.ops.delete(bm, geom=[f for f, s in zip(bm.faces, sel) if s == inv], context='FACES')
        bd = [e for e in bm.edges if e.is_boundary]
        if bd: bmesh.ops.holes_fill(bm, edges=bd, sides=0)
        bm.to_mesh(ob.data); bm.free(); ob.data.update()
    return o2

def ellipse_tube(name, centre, axis, rx, ry, length, mat):
    bm = bmesh.new(); r = bmesh.ops.create_cone(bm, cap_ends=True, segments=32, radius1=1, radius2=1, depth=length)
    bmesh.ops.scale(bm, verts=r['verts'], vec=(rx, ry, 1))
    q = V((0, 0, 1)).rotation_difference(V(axis)); bmesh.ops.rotate(bm, verts=r['verts'], cent=(0, 0, 0), matrix=q.to_matrix())
    bmesh.ops.translate(bm, verts=r['verts'], vec=V(centre))
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free(); me.materials.append(mat)
    for p in me.polygons: p.use_smooth = True
    o = O.new(name, me); sc.collection.objects.link(o); return o

# ---------------------------------------------------------------- split arms
ARM = {}
for side in ('arm_l', 'arm_r'):
    a = O[side]; v = np.array([x.co[:] for x in a.data.vertices]); sgn = 1 if v[:, 0].mean() > 0 else -1
    lo = v[(v[:, 2] > 0.395) & (v[:, 2] < 0.405)]; hi = v[(v[:, 2] > 0.485) & (v[:, 2] < 0.495)]
    c_lo = (lo.min(0) + lo.max(0)) / 2; c_hi = (hi.min(0) + hi.max(0)) / 2
    axis = V(c_hi - c_lo).normalized()
    pad = split(a, lambda c: c[:, 2] > Z_PAD, side + '__pad')
    up = split(a, lambda c: c[:, 2] > Z_TUBE, side + '__upper')
    tube = ellipse_tube(side + '__tube', V(c_lo) + axis * 0.035, axis, (lo[:, 0].max() - lo[:, 0].min()) / 2 * 0.92,
                        (lo[:, 1].max() - lo[:, 1].min()) / 2 * 0.92, 0.09, a.data.materials[0])
    pv = np.array([x.co[:] for x in pad.data.vertices]); pc = V(((pv.min(0) + pv.max(0)) / 2).tolist())
    ARM[side] = dict(sgn=sgn, axis=axis, sleeve=V(c_lo.tolist()) + axis * (Z_TUBE - c_lo[2]) / axis.z, pad=pc)
    print('ARM', side, 'axis', [round(x, 3) for x in axis], 'pad centre', [round(x, 3) for x in pc], flush=True)

# ---------------------------------------------------------------- headrest lock stays on the spine
h = O['headrest']; li = [i for i, m in enumerate(h.data.materials) if m and m.name == 'frame_plastic']
if li:
    mi = np.zeros(len(h.data.polygons), np.int32); h.data.polygons.foreach_get('material_index', mi)
    lock = split(h, lambda c, m=(mi == li[0]): m, 'frame__lock'); print('LOCK faces', int((mi == li[0]).sum()), flush=True)

# ---------------------------------------------------------------- controls
def empty(name, loc, parent=None, size=0.04, shape='PLAIN_AXES'):
    e = O.new(name, None); e.empty_display_type = shape; e.empty_display_size = size; sc.collection.objects.link(e)
    e.location = loc
    if parent: e.parent = parent; e.matrix_parent_inverse = parent.matrix_world.inverted()
    bpy.context.view_layer.update(); return e

def carry(ctrl, names):
    bpy.context.view_layer.update()
    for n in names:
        o = O[n]; o.parent = ctrl; o.matrix_parent_inverse = ctrl.matrix_world.inverted()

rig = empty('CHAIR_RIG', (0, 0, 0), size=0.25, shape='ARROWS')
PROPS = [('recline_deg', 0, 45, 'seat-to-back angle increase (deg): 0 = upright lock, 18 = 120 lock, 33 = 135 lock'),
         ('seat_slide', 0, 0.05, 'seat cushion forward (m)'), ('lumbar_depth', 0, 0.04, 'lumbar pad forward (m)'),
         ('arm_height', 0, 0.07, 'armrest up (m)'), ('arm_depth', -0.03, 0.03, 'armrest pad forward (m)'),
         ('arm_width', -0.02, 0.025, 'armrest pad outward (m)'), ('arm_rotate_deg', -20, 20, 'armrest pad turn (deg, front inward +)'),
         ('headrest_height', 0, 0.06, 'headrest up (m)')]
for k, lo_, hi_, desc in PROPS:
    rig[k] = 0.0; ui = rig.id_properties_ui(k); ui.update(min=lo_, max=hi_, soft_min=lo_, soft_max=hi_, description=desc)

def drive(obj, path, idx, expr, props):
    fc = obj.driver_add(path, idx); d = fc.driver; d.type = 'SCRIPTED'
    for p in props:
        var = d.variables.new(); var.name = p; var.type = 'SINGLE_PROP'
        var.targets[0].id = rig; var.targets[0].data_path = '["%s"]' % p
    d.expression = expr

c_back = empty('ctrl_recline_back', RPIVOT); c_seat = empty('ctrl_recline_seat', SPIVOT)
drive(c_back, 'rotation_euler', 0, '-radians(recline_deg) / %.3f' % (1 - SYNC), ['recline_deg'])
drive(c_seat, 'rotation_euler', 0, '-radians(recline_deg) * %.3f' % (SYNC / (1 - SYNC)), ['recline_deg'])
lv = np.array([x.co[:] for x in O['lumbar'].data.vertices]); c_lumb = empty('ctrl_lumbar', ((lv.min(0) + lv.max(0)) / 2).tolist(), c_back)
drive(c_lumb, 'location', 1, '%.4f - lumbar_depth' % c_lumb.location.y, ['lumbar_depth'])
hv = np.array([x.co[:] for x in O['headrest'].data.vertices]); c_head = empty('ctrl_headrest', ((hv.min(0) + hv.max(0)) / 2).tolist(), c_back)
drive(c_head, 'location', 2, '%.4f + headrest_height' % c_head.location.z, ['headrest_height'])
sv = np.array([x.co[:] for x in O['seat'].data.vertices]); c_slide = empty('ctrl_seat_slide', ((sv.min(0) + sv.max(0)) / 2).tolist(), c_seat)
drive(c_slide, 'location', 1, '%.4f - seat_slide' % c_slide.location.y, ['seat_slide'])
carry(c_back, ['backrest', 'frame', 'frame__lock']); carry(c_lumb, ['lumbar']); carry(c_head, ['headrest']); carry(c_slide, ['seat'])
for side, A in ARM.items():
    s = A['sgn']; ax = A['axis']
    c_up = empty('ctrl_%s_height' % side, A['sleeve'], c_seat)
    for i in range(3): drive(c_up, 'location', i, '%.4f + %.4f * arm_height' % (c_up.location[i], ax[i]), ['arm_height'])
    c_pad = empty('ctrl_%s_pad' % side, A['pad'], c_up)
    drive(c_pad, 'location', 0, '%.4f + %d * arm_width' % (c_pad.location.x, s), ['arm_width'])
    drive(c_pad, 'location', 1, '%.4f - arm_depth' % c_pad.location.y, ['arm_depth'])
    drive(c_pad, 'rotation_euler', 2, '%d * radians(arm_rotate_deg)' % (-s), ['arm_rotate_deg'])
    carry(c_seat, [side, side + '__tube']); carry(c_up, [side + '__upper']); carry(c_pad, [side + '__pad'])
    O[side + '__upper']['move_axis'] = list(ax); O[side + '__pad']['move_axis'] = list(ax)
    O[side + '__pad']['move_pivot'] = list(A['pad']); O[side + '__pad']['outward'] = s
for e in [o for o in O if o.type == 'EMPTY' and o is not rig and o.parent is None]:
    e.parent = rig; e.matrix_parent_inverse = rig.matrix_world.inverted()
O['lumbar']['move_axis'] = [0, -1, 0]; O['headrest']['move_axis'] = [0, 0, 1]; O['seat']['move_axis'] = [0, -1, 0]
txt = bpy.data.texts.new('RIG_README'); txt.write(open(__file__).read().split('import bpy')[0])

bpy.ops.wm.save_as_mainfile(filepath=dst, compress=True); print('saved', dst, flush=True)

# self-check on the saved file (drivers are only valid once the file is reloaded): every property at its max moves its parts
bpy.ops.wm.open_mainfile(filepath=dst); rig = bpy.data.objects['CHAIR_RIG']
for k, lo_, hi_, _ in PROPS: rig[k] = hi_
rig.update_tag(); dg = bpy.context.evaluated_depsgraph_get(); dg.update()
bad = [d.driver.expression for o in bpy.data.objects if o.animation_data for d in o.animation_data.drivers if not d.driver.is_valid]
moved = {o.name: round(o.evaluated_get(dg).matrix_world.translation.length, 4) for o in bpy.data.objects if o.type == 'MESH' and o.parent}
print('RIG invalid drivers:', bad, '| parts offset at max (m):', json.dumps(moved), flush=True)
