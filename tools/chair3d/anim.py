# True-3D exploded view of the Tripo chair model in Blender (Cycles, CPU).
# Usage (inside the sandbox, next to chair.glb, co.npy, lab.npy, parts.py):
#   python3 anim.py test   -> renders a few check frames
#   python3 anim.py full   -> renders every frame to frames/####.png (RGBA, shadow-catcher floor)
import bpy, sys, math, time, numpy as np, mathutils
from parts import label_components, PART_NAMES

MODE = sys.argv[-1]
FPS, N = 24, 115                  # 4.8 s
W, H = 960, 540
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
vpart = label_components(co, lab)[lab]
me = src.data
nf = len(me.polygons); lst = np.zeros(nf, int); me.polygons.foreach_get('loop_start', lst)
lv = np.zeros(len(me.loops), int); me.loops.foreach_get('vertex_index', lv)
fpart = vpart[lv[lst]]
lt = np.zeros(nf, int); me.polygons.foreach_get('loop_total', lt)
assert (lt == 3).all(), 'expects a triangle mesh'
li = lst[:, None] + np.arange(3)                    # loop indices of each triangle
tri = lv[li]
uv = np.zeros(len(me.loops) * 2, np.float32); me.uv_layers[0].data.foreach_get('uv', uv); uv = uv.reshape(-1, 2)[li]
objs = {}
for p, name in enumerate(PART_NAMES):
    sel = np.nonzero(fpart == p)[0]
    t = tri[sel]; used, inv = np.unique(t, return_inverse=True)
    m2 = bpy.data.meshes.new(name)
    m2.vertices.add(len(used)); m2.vertices.foreach_set('co', co[used].reshape(-1))       # world coordinates; objects keep identity transforms
    m2.loops.add(len(sel) * 3); m2.loops.foreach_set('vertex_index', inv.reshape(-1).astype(np.int32))
    m2.polygons.add(len(sel)); m2.polygons.foreach_set('loop_start', (np.arange(len(sel)) * 3).astype(np.int32))
    m2.uv_layers.new(); m2.uv_layers[0].data.foreach_set('uv', uv[sel].reshape(-1))
    m2.update(); m2.shade_smooth(); m2.materials.append(mat)
    o = bpy.data.objects.new(name, m2); bpy.context.scene.collection.objects.link(o); objs[name] = o
bpy.data.objects.remove(src)
print('parts', sorted(objs), flush=True)

# the AI texture reads mid-grey; darken it back to the real black mesh/plastic
nt = mat.node_tree; pb = [n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'][0]
if pb.inputs['Base Color'].links:
    lk = pb.inputs['Base Color'].links[0]; g = nt.nodes.new('ShaderNodeGamma'); g.inputs[1].default_value = 2.2
    nt.links.new(lk.from_socket, g.inputs[0]); nt.links.new(g.outputs[0], pb.inputs['Base Color'])
# black plastic/fabric: no metal, soft low sheen (the ORM map's metallic made it read grey)
chrome_src = mat.copy()
for l in list(pb.inputs['Metallic'].links): nt.links.remove(l)
pb.inputs['Metallic'].default_value = 0.0
pb.inputs['Specular IOR Level'].default_value = 0.3

# polished chrome for the base and gas lift (keeps the texture's dark details)
chrome = chrome_src; chrome.name = 'chrome'
bsdf = [n for n in chrome.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'][0]
for l in list(bsdf.inputs['Metallic'].links) + list(bsdf.inputs['Roughness'].links) + list(bsdf.inputs['Base Color'].links):
    chrome.node_tree.links.remove(l)
bsdf.inputs['Base Color'].default_value = (0.9, 0.9, 0.92, 1)
bsdf.inputs['Metallic'].default_value = 1.0; bsdf.inputs['Roughness'].default_value = 0.08
for n in ('base', 'gas_lift'):
    objs[n].data.materials[0] = chrome

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
ramp.color_ramp.elements[0].position = 0.45; ramp.color_ramp.elements[0].color = (0.04, 0.04, 0.045, 1)
ramp.color_ramp.elements[1].position = 0.62; ramp.color_ramp.elements[1].color = (0.95, 0.95, 0.97, 1)
wl.new(ramp.outputs['Color'], bg.inputs[0])

def area(name, loc, energy, size):
    L = bpy.data.lights.new(name, 'AREA'); L.energy = energy; L.size = size
    o = bpy.data.objects.new(name, L); sc.collection.objects.link(o); o.location = loc
    o.rotation_euler = (-mathutils.Vector(loc)).to_track_quat('-Z', 'Y').to_euler(); return o
area('key', (-1.2, -2.2, 2.2), 520, 2.5)
area('fill', (1.8, -0.8, 1.0), 140, 3.0)
area('rim', (-2.0, 1.8, 1.6), 450, 1.5)
bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, FLOOR0)); floor = bpy.context.object
floor.is_shadow_catcher = True

cam = bpy.data.cameras.new('cam'); cam.lens = 50
camo = bpy.data.objects.new('cam', cam); sc.collection.objects.link(camo); sc.camera = camo

def camera_at(t):
    k = smooth(t / 4.8)
    az = math.radians(228 + 24 * k)                     # slow orbit around the back three-quarter
    e = amount(3.5, t)                                  # pull back while the parts are apart
    dist = 3.0 + 1.8 * e; el = math.radians(10)
    target = mathutils.Vector((-0.08 - 0.08 * e, 0.0, 0.0))
    pos = target + dist * mathutils.Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
    camo.location = pos; camo.rotation_euler = (target - pos).to_track_quat('-Z', 'Y').to_euler()

def pose(t):
    for n, (i, off) in MOVES.items():
        k = amount(i, t); objs[n].location = (off[0] * k, off[1] * k, off[2] * k)
    floor.location.z = FLOOR0 - WHEEL_DROP * amount(7, t)
    camera_at(t)

import os
frames = [0, 60] if MODE == 'test' else range(int(os.environ.get('F0', 0)), int(os.environ.get('F1', N)) + 1); os.makedirs('frames', exist_ok=True)
for f in frames:
    t0 = time.time(); pose(f / FPS)
    sc.render.filepath = f'/home/user/frames/{f:04d}.png'; sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGBA'
    bpy.ops.render.render(write_still=True)
    print('frame', f, round(time.time() - t0, 1), flush=True)
