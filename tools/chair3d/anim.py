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
for _ in range(len(PART_NAMES) - 1): me.materials.append(mat)
me.polygons.foreach_set('material_index', fpart.astype(np.int32)); me.update()
bpy.context.view_layer.objects.active = src; src.select_set(True)
bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.separate(type='MATERIAL'); bpy.ops.object.mode_set(mode='OBJECT')
objs = {}
for o in bpy.context.scene.objects:
    if o.type != 'MESH': continue
    idx = [p.material_index for p in o.data.polygons[:1]][0]
    objs[PART_NAMES[idx]] = o; o.name = PART_NAMES[idx]
    bpy.context.view_layer.objects.active = o
    bpy.ops.object.material_slot_remove_unused()
print('parts', sorted(objs), flush=True)

# polished chrome for the base and gas lift (keeps the texture's dark details)
chrome = mat.copy(); chrome.name = 'chrome'
bsdf = [n for n in chrome.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'][0]
for l in list(bsdf.inputs['Metallic'].links) + list(bsdf.inputs['Roughness'].links): chrome.node_tree.links.remove(l)
bsdf.inputs['Metallic'].default_value = 1.0; bsdf.inputs['Roughness'].default_value = 0.12
for n in ('base', 'gas_lift'):
    objs[n].data.materials[0] = chrome

# wheel offsets: straight down plus outward from the hub
hub = np.array([0.01, 0.0])
for i in range(5):
    o = objs['wheel%d' % i]
    v = np.array([o.matrix_world @ p.co for p in o.data.vertices[::50]])
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
bg = world.node_tree.nodes['Background']; bg.inputs[0].default_value = (0.62, 0.62, 0.64, 1); bg.inputs[1].default_value = 0.9

def area(name, loc, energy, size):
    L = bpy.data.lights.new(name, 'AREA'); L.energy = energy; L.size = size
    o = bpy.data.objects.new(name, L); sc.collection.objects.link(o); o.location = loc
    o.rotation_euler = (-mathutils.Vector(loc)).to_track_quat('-Z', 'Y').to_euler(); return o
area('key', (-1.2, -2.2, 2.2), 900, 2.5)
area('fill', (1.8, -0.8, 1.0), 250, 3.0)
area('rim', (-2.0, 1.8, 1.6), 500, 1.5)
bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, FLOOR0)); floor = bpy.context.object
floor.is_shadow_catcher = True

cam = bpy.data.cameras.new('cam'); cam.lens = 50
camo = bpy.data.objects.new('cam', cam); sc.collection.objects.link(camo); sc.camera = camo

def camera_at(t):
    k = smooth(t / 4.8)
    az = math.radians(212 + 26 * k)                     # slow orbit around the back three-quarter
    e = amount(3.5, t)                                  # pull back while the parts are apart
    dist = 3.0 + 1.2 * e; el = math.radians(10)
    target = mathutils.Vector((-0.08, 0.0, 0.0 - 0.08 * e))
    pos = target + dist * mathutils.Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
    camo.location = pos; camo.rotation_euler = (target - pos).to_track_quat('-Z', 'Y').to_euler()

def pose(t):
    for n, (i, off) in MOVES.items():
        k = amount(i, t); objs[n].location = (off[0] * k, off[1] * k, off[2] * k)
    floor.location.z = FLOOR0 - WHEEL_DROP * amount(7, t)
    camera_at(t)

frames = [0, 30, 60, 90] if MODE == 'test' else range(N + 1)
import os; os.makedirs('frames', exist_ok=True)
for f in frames:
    t0 = time.time(); pose(f / FPS)
    sc.render.filepath = f'/home/user/frames/{f:04d}.png'; sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGBA'
    bpy.ops.render.render(write_still=True)
    print('frame', f, round(time.time() - t0, 1), flush=True)
