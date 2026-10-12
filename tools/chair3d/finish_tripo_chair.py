# After fit_tripo_parts.py: make the Tripo parts read like the real chair and complete the part set for anim3.py.
#  - Tripo materials renamed tp_<part> (anim3 leaves tp_* alone): base colour darkened toward the real black, metal kept
#    only on the base (chrome legs) and frame (chrome ring); other parts dielectric, slightly rough.
#  - gas lift rebuilt as clean bevelled cylinders (black body + dust-cover ring), from the base hub to the mechanism.
#  - base moved so its hub column sits exactly under the gas lift (the fit leaves it ~3 cm off).
#  - upper spine lifted to the backrest top (SPINE_DZ); headrest moved forward onto the spine's top clamp (HEAD_DY, HEAD_DZ).
#  - wheel_0..4: the Tripo base already includes its castors; anim3 still needs the objects -> tiny hidden stand-ins
#    at the old castor positions (they only steer the castors' fly-in direction, which now travels with the base).
# Usage: python finish_tripo_chair.py in.blend out.blend   (env: DARK base-colour factor for non-metal, 0.42)
import bpy, bmesh, sys, os, numpy as np, mathutils
src, dst = sys.argv[-2:]
bpy.ops.wm.open_mainfile(filepath=src)
DARK = float(os.environ.get('DARK', 0.42)); KEEP_METAL = {'base', 'frame'}
WHEEL_POS = {'wheel_0': (0.002, 0.198, 0.024), 'wheel_1': (-0.149, -0.226, 0.024), 'wheel_2': (0.132, -0.246, 0.024),
             'wheel_3': (-0.228, 0.030, 0.024), 'wheel_4': (0.233, -0.015, 0.024)}

for o in [o for o in bpy.data.objects if o.type == 'MESH']:
    for m in o.data.materials:
        if not m or not m.name.startswith('tripo_material'): continue
        m.name = 'tp_' + o.name; nt = m.node_tree; L = nt.links
        pb = [n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'][0]
        bc = pb.inputs['Base Color'].links[0].from_socket if pb.inputs['Base Color'].links else None
        met = pb.inputs['Metallic'].links[0].from_socket if pb.inputs['Metallic'].links else None
        keep = o.name in KEEP_METAL and met is not None
        if not keep:
            for l in list(pb.inputs['Metallic'].links): L.remove(l)
            pb.inputs['Metallic'].default_value = 0.0
        if bc is not None:
            fac = nt.nodes.new('ShaderNodeMapRange'); fac.inputs['To Min'].default_value = DARK; fac.inputs['To Max'].default_value = 1.0
            if keep: L.new(met, fac.inputs['Value'])
            else: fac.inputs['Value'].default_value = 0.0
            mul = nt.nodes.new('ShaderNodeMix'); mul.data_type = 'RGBA'; mul.blend_type = 'MULTIPLY'; mul.inputs['Factor'].default_value = 1.0
            L.new(bc, mul.inputs[6]); L.new(fac.outputs['Result'], mul.inputs[7]); L.new(mul.outputs[2], pb.inputs['Base Color'])
        if not keep and pb.inputs['Roughness'].links:     # keep Tripo's roughness detail but never glossy plastic
            r = pb.inputs['Roughness'].links[0].from_socket
            mx = nt.nodes.new('ShaderNodeMath'); mx.operation = 'MAXIMUM'; mx.inputs[1].default_value = 0.5
            L.new(r, mx.inputs[0]); L.new(mx.outputs[0], pb.inputs['Roughness'])
        print('MAT', o.name, m.name, 'metal kept' if keep else 'dielectric')

# ---- gas lift
old = bpy.data.objects['gas_lift']; v = np.array([old.matrix_world @ x.co for x in old.data.vertices])
cx, cy = (v[:, 0].min() + v[:, 0].max()) / 2, (v[:, 1].min() + v[:, 1].max()) / 2
mat = bpy.data.materials.get('frame_plastic'); bpy.data.objects.remove(old, do_unlink=True)
# the fitted Tripo base lands ~3 cm off the seat axis: move it so its hub column (top 4 cm) is centred under the lift
base = bpy.data.objects['base']; bv = np.array([base.matrix_world @ x.co for x in base.data.vertices]); top = bv[bv[:, 2] > bv[:, 2].max() - 0.04]
SHIFT = np.array([cx, cy]) - (top[:, :2].min(0) + top[:, :2].max(0)) / 2
base.data.transform(mathutils.Matrix.Translation((SHIFT[0], SHIFT[1], 0))); base.data.update(); print('BASE SHIFT', SHIFT.round(4))
bv = np.array([base.matrix_world @ x.co for x in base.data.vertices]); near = np.hypot(bv[:, 0] - cx, bv[:, 1] - cy) < 0.03
z0 = float(bv[near, 2].max()) - 0.03 if near.any() else 0.13; z1 = 0.305
bm = bmesh.new()
def cyl(r, za, zb, seg=64):
    res = bmesh.ops.create_cone(bm, cap_ends=True, segments=seg, radius1=r, radius2=r, depth=zb - za)
    bmesh.ops.translate(bm, verts=res['verts'], vec=(cx, cy, (za + zb) / 2))
cyl(0.0255, z0, z1 - 0.07)            # body
cyl(0.0270, z1 - 0.075, z1 - 0.06)    # dust-cover ring
cyl(0.0175, z1 - 0.06, z1)            # upper tube into the mechanism
me = bpy.data.meshes.new('gas_lift'); bm.to_mesh(me); bm.free()
gl = bpy.data.objects.new('gas_lift', me); bpy.context.scene.collection.objects.link(gl)
if mat: me.materials.append(mat)
bv_ = gl.modifiers.new('bev', 'BEVEL'); bv_.width = 0.0015; bv_.segments = 3; bv_.limit_method = 'ANGLE'
bpy.context.view_layer.objects.active = gl; bpy.ops.object.modifier_apply(modifier='bev')
for p in me.polygons: p.use_smooth = True
print('GAS', round(z0, 3), z1, round(cx, 4), round(cy, 4))

# ---- spine top ends ~2 cm below the backrest top; on the real chair both meet at one height under the headrest.
# Lift only the upper spine (smooth ramp from SPINE_Z0 up), so the seat and lumbar attachments stay put.
fr = bpy.data.objects['frame']; SPINE_DZ = float(os.environ.get('SPINE_DZ', 0.02)); SPINE_Z0 = float(os.environ.get('SPINE_Z0', 0.62))
fz1 = max(v.co.z for v in fr.data.vertices)
for v in fr.data.vertices:
    t = min(max((v.co.z - SPINE_Z0) / (fz1 - SPINE_Z0), 0), 1); v.co.z += SPINE_DZ * t * t * (3 - 2 * t)
fr.data.update(); print('SPINE TOP', round(fz1, 3), '->', round(fz1 + SPINE_DZ, 3))

# ---- headrest (old model) stands ~10 mm behind the Tripo spine's top clamp: seat its post against the clamp like the real chair
hr = bpy.data.objects['headrest']; HEAD_DY = float(os.environ.get('HEAD_DY', -0.012)); HEAD_DZ = float(os.environ.get('HEAD_DZ', -0.015))
hr.data.transform(mathutils.Matrix.Translation((0, HEAD_DY, HEAD_DZ))); hr.data.update(); print('HEAD SHIFT', HEAD_DY, HEAD_DZ)

# ---- castor stand-ins
for n, (x, y, z) in WHEEL_POS.items():
    x, y = x + SHIFT[0], y + SHIFT[1]
    me = bpy.data.meshes.new(n); me.from_pydata([(x, y, z), (x + 1e-4, y, z), (x, y + 1e-4, z)], [], [(0, 1, 2)])
    o = bpy.data.objects.new(n, me); bpy.context.scene.collection.objects.link(o); o.hide_render = True
bpy.ops.wm.save_as_mainfile(filepath=dst); print('saved', dst)
