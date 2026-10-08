# Turntable exploded view (the Tripo "Explosion" look): the chair turns a full 360 degrees while its parts
# fly far apart into a vertical stack (headrest on top, base and wheels at the bottom, armrests out to the
# sides), hold, and come back together. The last frame is identical to the first.
# Uses chair_fixed.blend from fix_m8.py (or fix_m4.py / tripo_fix.py).
# Defaults match the real 3/4 product photo (kf1B): angle, deep blacks, soft shadow under the chair.
# Usage: python3 anim3.py test|full   (env: BLEND, RES, SAMPLES, FRAMES, F0, F1, LIGHT, TEX_GAMMA, DURATION)
import bpy, sys, os, math, time, numpy as np, mathutils

MODE = sys.argv[-1]
FPS = 24
SECONDS = float(os.environ.get('DURATION', 4.8))     # (not SECONDS: bash owns that name)
N = int(round(SECONDS * FPS)) - 1          # last frame index; frames 0..N
W = int(os.environ.get('RES', 960)); H = W * 9 // 16
SAMPLES = int(os.environ.get('SAMPLES', 16))
# part -> (stagger order, offset in metres in the chair's own frame: x forward, y lateral, z up)
MOVES = {
    'headrest':  (0, (0.00, 0.0, 0.62)),
    'backrest':  (1, (-0.04, 0.0, 0.40)),
    'frame':     (1, (-0.24, 0.0, 0.20)),
    'lumbar':    (2, (0.10, 0.0, 0.17)),
    'arm_r':     (2, (0.00, 0.33, 0.10)),
    'arm_l':     (2, (0.00, -0.33, 0.10)),
    'seat':      (3, (0.00, 0.0, 0.00)),
    'mechanism': (4, (0.00, 0.0, -0.17)),
    'gas_lift':  (5, (0.00, 0.0, -0.31)),
    'base':      (6, (0.00, 0.0, -0.46)),
}
WHEEL_DROP, WHEEL_OUT = 0.60, 0.10
# timeline as fractions of the clip: assembled, explode, hold, return, assembled
T_OUT, T_HOLD, T_BACK, T_END = 0.10, 0.32, 0.68, 0.90
STAG = 0.012                                # stagger between parts (fraction of the clip)
FLOOR0 = -0.4904

def smooth(u):
    u = min(max(u, 0.0), 1.0); return u * u * (3 - 2 * u)

EXPLODE = os.environ.get('EXPLODE', '1') == '1'        # EXPLODE=0: plain 360 turntable of the assembled chair (USP shot)

def amount(i, u):
    if not EXPLODE: return 0.0
    a = smooth((u - (T_OUT + i * STAG)) / (T_HOLD - T_OUT - 6 * STAG))
    b = smooth((u - (T_BACK + (7 - i) * STAG)) / (T_END - T_BACK - 6 * STAG))
    return a * (1 - b)

bpy.ops.wm.open_mainfile(filepath=os.environ.get('BLEND', 'chair_fixed.blend'))
sc = bpy.context.scene
objs = {o.name.replace('wheel_', 'wheel'): o for o in sc.objects if o.type == 'MESH'}
R = mathutils.Matrix.Translation((0, 0, FLOOR0)) @ mathutils.Matrix.Rotation(math.pi / 2, 4, 'Z')
for o in objs.values():
    o.data.transform(R); o.data.update()

# ---- materials (same treatment as anim2.py)
def tripo_colour(m):
    nt = m.node_tree; pb = [n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'][0]
    lk = pb.inputs['Base Color'].links
    if not lk: return
    src = lk[0].from_socket
    hsv = nt.nodes.new('ShaderNodeHueSaturation'); hsv.inputs['Saturation'].default_value = float(os.environ.get('TEX_SAT', 0.3))
    gm = nt.nodes.new('ShaderNodeGamma'); gm.inputs[1].default_value = float(os.environ.get('TEX_GAMMA', 1.2))
    nt.links.new(src, hsv.inputs['Color']); nt.links.new(hsv.outputs['Color'], gm.inputs[0]); nt.links.new(gm.outputs[0], pb.inputs['Base Color'])
    for l in list(pb.inputs['Metallic'].links): nt.links.remove(l)
    pb.inputs['Metallic'].default_value = 0.0
    pb.inputs['Roughness'].default_value = max(pb.inputs['Roughness'].default_value, 0.6)

mats = {m for o in objs.values() for m in o.data.materials}
for m in mats:
    if m.name.endswith('fabric') or m.name == 'frame_plastic': continue
    tripo_colour(m)

def add_glow(m):
    # two effects while a part breaks away / locks back in:
    #  - a thin light ray: a sharp cool-white line that sweeps across the part (bottom to top as it leaves,
    #    top to bottom as it returns), driven by the object's 'sweep' (position) and 'ray' (brightness)
    #  - a faint rim glow on the part's silhouette ('glow')
    nt = m.node_tree; out = [n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'][0]; L = nt.links
    surf = out.inputs['Surface'].links[0].from_socket
    def node(t, **kw):
        n = nt.nodes.new(t)
        for k, v in kw.items(): setattr(n, k, v)
        return n
    def attr(name):
        a = node('ShaderNodeAttribute', attribute_type='OBJECT', attribute_name=name); return a.outputs['Fac']
    def math(op, a, b):
        n = node('ShaderNodeMath', operation=op)
        for i, x in enumerate((a, b)):
            if isinstance(x, (int, float)): n.inputs[i].default_value = x
            else: L.new(x, n.inputs[i])
        return n.outputs[0]
    tc = node('ShaderNodeTexCoord'); sep = node('ShaderNodeSeparateXYZ'); L.new(tc.outputs['Generated'], sep.inputs[0])
    pos = math('ADD', math('MULTIPLY', sep.outputs['Z'], 0.8), math('MULTIPLY', sep.outputs['X'], 0.2))   # slightly diagonal line
    d = math('ABSOLUTE', math('SUBTRACT', pos, attr('sweep')), 0)
    mr = node('ShaderNodeMapRange', interpolation_type='SMOOTHSTEP'); L.new(d, mr.inputs['Value'])
    mr.inputs['From Min'].default_value = 0.0; mr.inputs['From Max'].default_value = float(os.environ.get('RAY_W', 0.018))
    mr.inputs['To Min'].default_value = 1.0; mr.inputs['To Max'].default_value = 0.0
    ray = math('MULTIPLY', math('MULTIPLY', mr.outputs['Result'], attr('ray')), float(os.environ.get('RAY', 6.0)))
    # on flat, level faces (seat underside, base top) the band would light the whole face at once: keep it on the sides
    geo = node('ShaderNodeNewGeometry'); gn = node('ShaderNodeSeparateXYZ'); L.new(geo.outputs['Normal'], gn.inputs[0])
    lv = node('ShaderNodeMapRange', interpolation_type='SMOOTHSTEP'); L.new(math('ABSOLUTE', gn.outputs['Z'], 0), lv.inputs['Value'])
    lv.inputs['From Min'].default_value = 0.55; lv.inputs['From Max'].default_value = 0.85
    lv.inputs['To Min'].default_value = 1.0; lv.inputs['To Max'].default_value = 0.0
    ray = math('MULTIPLY', ray, lv.outputs['Result'])
    lw = node('ShaderNodeLayerWeight'); lw.inputs['Blend'].default_value = 0.2
    rim = math('MULTIPLY', math('MULTIPLY', math('POWER', lw.outputs['Facing'], float(os.environ.get('GLOW_POW', 7.0))), attr('glow')),
               float(os.environ.get('GLOW', 0.6)))
    em = node('ShaderNodeEmission'); em.inputs['Color'].default_value = (0.85, 0.93, 1.0, 1)
    L.new(math('ADD', ray, rim), em.inputs['Strength'])
    add = node('ShaderNodeAddShader'); L.new(surf, add.inputs[0]); L.new(em.outputs[0], add.inputs[1]); L.new(add.outputs[0], out.inputs['Surface'])
for m in mats: add_glow(m)

def centre(o):
    v = np.zeros(len(o.data.vertices) * 3); o.data.vertices.foreach_get('co', v); v = v.reshape(-1, 3); return (v.min(0) + v.max(0)) / 2

# ---- turntable: every part hangs off a pivot on the gas-lift axis; offsets are in the chair's own frame
hub = centre(objs['gas_lift'])[:2]
pivot = bpy.data.objects.new('pivot', None); sc.collection.objects.link(pivot); pivot.location = (hub[0], hub[1], 0)
for o in objs.values():
    o.data.transform(mathutils.Matrix.Translation((-hub[0], -hub[1], 0))); o.parent = pivot; o.matrix_parent_inverse.identity()
for i in range(5):
    d = centre(objs['wheel%d' % i])[:2]; d /= np.linalg.norm(d)
    MOVES['wheel%d' % i] = (7, (WHEEL_OUT * d[0], WHEEL_OUT * d[1], -WHEEL_DROP))

# ---- scene
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = SAMPLES
sc.cycles.use_denoising = True; sc.render.film_transparent = True
sc.render.resolution_x, sc.render.resolution_y = W, H; sc.render.fps = FPS
sc.render.use_persistent_data = True; sc.view_settings.view_transform = 'Standard'
LIGHT = float(os.environ.get('LIGHT', 0.09))
world = bpy.data.worlds.new('w'); sc.world = world; world.use_nodes = True
bg = world.node_tree.nodes['Background']; bg.inputs[0].default_value = (0.62, 0.62, 0.64, 1); bg.inputs[1].default_value = 0.45 * LIGHT

# chrome needs something bright to reflect: glossy rays see the full-strength studio surround
_wn, _wl = world.node_tree.nodes, world.node_tree.links
_out = [n for n in _wn if n.type == 'OUTPUT_WORLD'][0]
_bg2 = _wn.new('ShaderNodeBackground'); _bg2.inputs[0].default_value = (0.62, 0.62, 0.64, 1); _bg2.inputs[1].default_value = float(os.environ.get('CHROME_ENV', 0.3))
_lp = _wn.new('ShaderNodeLightPath'); _mx = _wn.new('ShaderNodeMixShader')
_wl.new(_lp.outputs['Is Glossy Ray'], _mx.inputs[0]); _wl.new(bg.outputs[0], _mx.inputs[1]); _wl.new(_bg2.outputs[0], _mx.inputs[2]); _wl.new(_mx.outputs[0], _out.inputs['Surface'])

LROT = math.radians(float(os.environ.get('LIGHT_ROT', 23)))      # turn the light rig with the camera (CAM_AZ - 15)
def area(name, loc, energy, size):
    loc = (loc[0] * math.cos(LROT) - loc[1] * math.sin(LROT), loc[0] * math.sin(LROT) + loc[1] * math.cos(LROT), loc[2])
    L = bpy.data.lights.new(name, 'AREA'); L.energy = energy * LIGHT; L.size = size
    o = bpy.data.objects.new(name, L); sc.collection.objects.link(o); o.location = loc
    o.rotation_euler = (-mathutils.Vector(loc)).to_track_quat('-Z', 'Y').to_euler(); return o
area('key', (2.2, -1.4, 2.2), 520, float(os.environ.get('KEY_SIZE', 6)))       # front-left key, front fill, back rim: the camera looks from the front
area('fill', (1.6, 1.8, 0.8), 160, 3.0)
area('rim', (-2.2, 0.6, 1.8), 450, 1.5)
bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 0, FLOOR0)); floor = bpy.context.object
floor.is_shadow_catcher = True; floor.visible_glossy = False

cam = bpy.data.cameras.new('cam'); cam.lens = 50
camo = bpy.data.objects.new('cam', cam); sc.collection.objects.link(camo); sc.camera = camo
AZ = math.radians(float(os.environ.get('CAM_AZ', 38)))     # a little off straight-front
EL = math.radians(float(os.environ.get('CAM_EL', 10)))
D0, D1 = float(os.environ.get('D0', 3.1)), float(os.environ.get('D1', 7.0))

def camera_at(e):
    dist = D0 + (D1 - D0) * e
    target = mathutils.Vector((hub[0], hub[1], float(os.environ.get('CAM_Z', 0.02)) - 0.04 * e))
    pos = target + dist * mathutils.Vector((math.cos(EL) * math.cos(AZ), math.cos(EL) * math.sin(AZ), math.sin(EL)))
    camo.location = pos; camo.rotation_euler = (target - pos).to_track_quat('-Z', 'Y').to_euler()

def pose(f):
    u = f / N
    pivot.rotation_euler = (0, 0, 2 * math.pi * u)                    # one full turn: frame N = frame 0
    for n, (i, off) in MOVES.items():
        k = amount(i, u); objs[n].location = (off[0] * k, off[1] * k, off[2] * k); objs[n]['glow'] = 4 * k * (1 - k)
        objs[n]['sweep'] = -0.06 + 1.12 * k; objs[n]['ray'] = min(1.0, 8 * k * (1 - k))
    e = smooth((u - T_OUT + 0.03) / (T_HOLD - T_OUT)) * (1 - smooth((u - T_BACK) / (T_END - T_BACK + 0.03)))
    if not EXPLODE: e = 0.0
    floor.location.z = FLOOR0 - WHEEL_DROP * amount(7, u)
    camera_at(e)
    bpy.context.view_layer.update()

if MODE == 'parts':
    pose(0); floor.hide_render = True
    sc.render.resolution_x = sc.render.resolution_y = int(os.environ.get('PRES', 700))
    os.makedirs('parts', exist_ok=True)
    order = os.environ.get('ONLY', 'headrest,backrest,frame,lumbar,seat,arm_l,arm_r,mechanism,gas_lift,base,wheel0').split(',')
    views = [(f'v{i}', float(v.split('/')[0]), float(v.split('/')[1])) for i, v in enumerate(os.environ.get('VIEWS', '15/8,195/8,75/12').split(','))]
    for n in order:
        for o in objs.values(): o.hide_render = (o is not objs[n])
        mw = objs[n].matrix_world; vv = [mw @ mathutils.Vector(c) for c in objs[n].bound_box]
        lo = mathutils.Vector([min(p[i] for p in vv) for i in range(3)]); hi = mathutils.Vector([max(p[i] for p in vv) for i in range(3)])
        c = (lo + hi) / 2; r = (hi - lo).length / 2
        for side, azd, eld in views:
            az, el = math.radians(azd), math.radians(eld); pos = c + 3.4 * r * mathutils.Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
            camo.location = pos; camo.rotation_euler = (c - pos).to_track_quat('-Z', 'Y').to_euler(); cam.lens = 50
            sc.render.filepath = f'{os.getcwd()}/parts/{n}_{side}.png'; bpy.ops.render.render(write_still=True)
        print('part', n, flush=True)
    sys.exit()

for _n in filter(None, os.environ.get('HIDE', '').split(',')): objs[_n].hide_render = True    # debugging aid
frames = [int(f) for f in os.environ.get('FRAMES', '0,30,57').split(',')] if MODE == 'test' else range(int(os.environ.get('F0', 0)), int(os.environ.get('F1', N)) + 1)
os.makedirs('frames', exist_ok=True)
for f in frames:
    t0 = time.time(); pose(f)
    sc.render.filepath = f'{os.getcwd()}/frames/{f:04d}.png'; sc.render.image_settings.file_format = 'PNG'; sc.render.image_settings.color_mode = 'RGBA'
    bpy.ops.render.render(write_still=True)
    print('frame', f, round(time.time() - t0, 1), flush=True)
