# Exploded view of the Tripo Studio chair (already split and cleaned by tripo_fix.py into chair_fixed.blend).
# Same timing, camera, lights and glow as anim.py.
# Usage: python3 anim2.py test|parts|full   (env: BLEND, RES, SAMPLES, F0, F1, FRAMES, ONLY, PRES, VIEWS,
#        TEX_GAMMA / TEX_GAIN / TEX_SAT: colour of Tripo's textures, FAB_GAIN: real-fabric brightness)
import bpy, sys, os, math, time, numpy as np, mathutils

MODE = sys.argv[-1]
FPS, N = 24, 115                  # 4.8 s
W = int(os.environ.get('RES', 960)); H = W * 9 // 16
SAMPLES = int(os.environ.get('SAMPLES', 16))
MOVES = {                         # part -> (stagger order, offset in metres: x forward, y lateral, z up)
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
T0, STEP, DUR = 0.3, 0.15, 0.6
HOLD_END = 2.6
FLOOR0 = -0.4904

def smooth(u):
    u = min(max(u, 0.0), 1.0); return u * u * (3 - 2 * u)

def amount(i, t):
    return smooth((t - (T0 + i * STEP)) / DUR) * (1 - smooth((t - (HOLD_END + (7 - i) * STEP)) / DUR))

bpy.ops.wm.open_mainfile(filepath=os.environ.get('BLEND', 'chair_fixed.blend'))
sc = bpy.context.scene
objs = {o.name.replace('wheel_', 'wheel'): o for o in sc.objects if o.type == 'MESH'}
# into the working frame: front towards +x (the model faces -y), height centred on z = 0
R = mathutils.Matrix.Translation((0, 0, FLOOR0)) @ mathutils.Matrix.Rotation(math.pi / 2, 4, 'Z')
for o in objs.values():
    o.data.transform(R); o.data.update()
print('parts', sorted(objs), flush=True)

# ---- materials
def tripo_colour(m):
    # Tripo's de-lit texture reads slightly blue-grey: pull it to the real chair's neutral black
    nt = m.node_tree; pb = [n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'][0]
    lk = pb.inputs['Base Color'].links
    if not lk: return
    src = lk[0].from_socket
    hsv = nt.nodes.new('ShaderNodeHueSaturation'); hsv.inputs['Saturation'].default_value = float(os.environ.get('TEX_SAT', 0.3))
    hsv.inputs['Value'].default_value = float(os.environ.get('TEX_GAIN', 1.0))
    gm = nt.nodes.new('ShaderNodeGamma'); gm.inputs[1].default_value = float(os.environ.get('TEX_GAMMA', 1.0))
    nt.links.new(src, hsv.inputs['Color']); nt.links.new(hsv.outputs['Color'], gm.inputs[0]); nt.links.new(gm.outputs[0], pb.inputs['Base Color'])
    for l in list(pb.inputs['Metallic'].links): nt.links.remove(l)
    pb.inputs['Metallic'].default_value = 0.0
    pb.inputs['Roughness'].default_value = max(pb.inputs['Roughness'].default_value, 0.6)

mats = {m for o in objs.values() for m in o.data.materials}
for m in (mats if not os.environ.get('NOMAT') else []):
    if m.name.endswith('fabric'):
        g = [n for n in m.node_tree.nodes if n.type == 'GAMMA'][0]
        hv = m.node_tree.nodes.new('ShaderNodeHueSaturation'); hv.inputs['Value'].default_value = float(os.environ.get('FAB_GAIN', 1.0))
        hv.inputs['Saturation'].default_value = 0.3
        pb = [n for n in m.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'][0]
        m.node_tree.links.new(g.outputs[0], hv.inputs['Color']); m.node_tree.links.new(hv.outputs['Color'], pb.inputs['Base Color'])
    elif m.name == 'frame_plastic':
        pb = m.node_tree.nodes['Principled BSDF']; c = pb.inputs['Base Color'].default_value
        v = float(os.environ.get('PLASTIC', 0.022)); pb.inputs['Base Color'].default_value = (v, v, v * 1.05, 1)
    else:
        tripo_colour(m)

if os.environ.get('DEBUG_PLAIN'):
    _pm = bpy.data.materials.new('plain'); _pm.use_nodes = True; _pb = _pm.node_tree.nodes['Principled BSDF']
    _pb.inputs['Base Color'].default_value = (0.02, 0.02, 0.02, 1); _pb.inputs['Specular IOR Level'].default_value = float(os.environ['DEBUG_PLAIN'])
    for o in objs.values():
        for i in range(len(o.data.materials)): o.data.materials[i] = _pm
    mats = {_pm}

def add_glow(m):
    nt = m.node_tree; out = [n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'][0]
    surf = out.inputs['Surface'].links[0].from_socket
    lw = nt.nodes.new('ShaderNodeLayerWeight'); lw.inputs['Blend'].default_value = 0.2
    pw = nt.nodes.new('ShaderNodeMath'); pw.operation = 'POWER'; pw.inputs[1].default_value = 4.0
    at = nt.nodes.new('ShaderNodeAttribute'); at.attribute_type = 'OBJECT'; at.attribute_name = 'glow'
    mu = nt.nodes.new('ShaderNodeMath'); mu.operation = 'MULTIPLY'
    mu2 = nt.nodes.new('ShaderNodeMath'); mu2.operation = 'MULTIPLY'; mu2.inputs[1].default_value = 7.0
    em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Color'].default_value = (0.82, 0.92, 1.0, 1)
    add = nt.nodes.new('ShaderNodeAddShader'); L = nt.links
    L.new(lw.outputs['Facing'], pw.inputs[0]); L.new(pw.outputs[0], mu.inputs[0]); L.new(at.outputs['Fac'], mu.inputs[1])
    L.new(mu.outputs[0], mu2.inputs[0]); L.new(mu2.outputs[0], em.inputs['Strength'])
    L.new(surf, add.inputs[0]); L.new(em.outputs[0], add.inputs[1]); L.new(add.outputs[0], out.inputs['Surface'])
for m in (mats if not os.environ.get('NOGLOW') else []): add_glow(m)

def centre(o):
    v = np.zeros(len(o.data.vertices) * 3); o.data.vertices.foreach_get('co', v); v = v.reshape(-1, 3); return (v.min(0) + v.max(0)) / 2

hub = centre(objs['gas_lift'])[:2]
for i in range(5):
    d = centre(objs['wheel%d' % i])[:2] - hub; d /= np.linalg.norm(d)
    MOVES['wheel%d' % i] = (7, (WHEEL_OUT * d[0], WHEEL_OUT * d[1], -WHEEL_DROP))

# ---- scene: camera, lights, shadow-catcher floor that follows the wheels
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = SAMPLES
sc.cycles.use_denoising = True; sc.render.film_transparent = True
sc.render.resolution_x, sc.render.resolution_y = W, H; sc.render.fps = FPS
sc.render.use_persistent_data = True; sc.view_settings.view_transform = 'Standard'
world = bpy.data.worlds.new('w'); sc.world = world; world.use_nodes = True
LIGHT = float(os.environ.get('LIGHT', 1.0))         # overall light level (calibrated against the real photos)
bg = world.node_tree.nodes['Background']; bg.inputs[0].default_value = (0.62, 0.62, 0.64, 1); bg.inputs[1].default_value = float(os.environ.get('WORLD', 0.45)) * LIGHT

def area(name, loc, energy, size):
    L = bpy.data.lights.new(name, 'AREA'); L.energy = energy * LIGHT; L.size = size
    o = bpy.data.objects.new(name, L); sc.collection.objects.link(o); o.location = loc
    o.rotation_euler = (-mathutils.Vector(loc)).to_track_quat('-Z', 'Y').to_euler(); return o
area('key', (-1.2, -2.2, 2.2), 520, 2.5)
area('fill', (1.8, -0.8, 1.0), 140, 3.0)
area('rim', (-2.0, 1.8, 1.6), 450, 1.5)
bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, FLOOR0)); floor = bpy.context.object
floor.is_shadow_catcher = True; floor.visible_glossy = False

cam = bpy.data.cameras.new('cam'); cam.lens = 50
camo = bpy.data.objects.new('cam', cam); sc.collection.objects.link(camo); sc.camera = camo

def camera_at(t):
    az = math.radians(228 + 22 * math.sin(math.pi * t / 4.8))
    e = smooth((t - T0) / 0.8) * (1 - smooth((t - (HOLD_END + 7 * STEP)) / 0.8))
    dist = 3.0 + float(os.environ.get('CAM_PULL', 1.1)) * e; el = math.radians(10)
    target = mathutils.Vector((-0.08 - 0.08 * e, 0.0, float(os.environ.get('CAM_RISE', 0.06)) * e))
    pos = target + dist * mathutils.Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
    camo.location = pos; camo.rotation_euler = (target - pos).to_track_quat('-Z', 'Y').to_euler()

def pose(t):
    for n, (i, off) in MOVES.items():
        k = amount(i, t); objs[n].location = (off[0] * k, off[1] * k, off[2] * k); objs[n]['glow'] = 4 * k * (1 - k)
    floor.location.z = FLOOR0 - WHEEL_DROP * amount(7, t)
    camera_at(t)

if MODE == 'parts':
    pose(0.0); floor.hide_render = True
    sc.render.resolution_x = sc.render.resolution_y = int(os.environ.get('PRES', 300))
    os.makedirs('parts', exist_ok=True)
    order = os.environ.get('ONLY', 'headrest,backrest,frame,lumbar,seat,arm_l,arm_r,mechanism,gas_lift,base,wheel0').split(',')
    views = [('a', 228, 15), ('b', 40, 15)] if not os.environ.get('VIEWS') else [(f'v{i}', float(s.split('/')[0]), float(s.split('/')[1])) for i, s in enumerate(os.environ['VIEWS'].split(','))]
    for n in order:
        for o in objs.values(): o.hide_render = (o is not objs[n])
        c = mathutils.Vector(centre(objs[n]).tolist())
        v = np.zeros(len(objs[n].data.vertices) * 3); objs[n].data.vertices.foreach_get('co', v); v = v.reshape(-1, 3)
        r = float(np.linalg.norm(np.ptp(v, 0))) / 2
        for side, azd, eld in views:
            az, el = math.radians(azd), math.radians(eld); pos = c + 3.1 * r * mathutils.Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
            camo.location = pos; camo.rotation_euler = (c - pos).to_track_quat('-Z', 'Y').to_euler()
            sc.render.filepath = f'{os.getcwd()}/parts/{n}_{side}.png'; bpy.ops.render.render(write_still=True)
        print('part', n, flush=True)
    sys.exit()

frames = [int(f) for f in os.environ.get('FRAMES', '0,60').split(',')] if MODE == 'test' else range(int(os.environ.get('F0', 0)), int(os.environ.get('F1', N)) + 1)
os.makedirs('frames', exist_ok=True)
for f in frames:
    t0 = time.time(); pose(f / FPS)
    sc.render.filepath = f'{os.getcwd()}/frames/{f:04d}.png'; sc.render.image_settings.file_format = 'PNG'; sc.render.image_settings.color_mode = 'RGBA'
    bpy.ops.render.render(write_still=True)
    print('frame', f, round(time.time() - t0, 1), flush=True)
