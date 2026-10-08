# True-3D exploded view of the Tripo chair model in Blender (Cycles, CPU).
# Usage (inside the sandbox, next to chair.glb, co.npy, lab.npy, parts.py):
#   python3 anim.py test   -> renders a few check frames
#   python3 anim.py full   -> renders every frame to frames/####.png (RGBA, shadow-catcher floor)
import bpy, sys, math, time, numpy as np, mathutils
from parts import vertex_parts, PART_NAMES

MODE = sys.argv[-1]
FPS, N = 24, 115                  # 4.8 s
import os
W = int(os.environ.get('RES', 960)); H = W * 9 // 16
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
vpart = vertex_parts(co, lab, np.load('tri.npy'))
me = src.data
nf = len(me.polygons); lst = np.zeros(nf, int); me.polygons.foreach_get('loop_start', lst)
lv = np.zeros(len(me.loops), int); me.loops.foreach_get('vertex_index', lv)
fpart = vpart[lv[lst]]
lt = np.zeros(nf, int); me.polygons.foreach_get('loop_total', lt)
assert (lt == 3).all(), 'expects a triangle mesh'
li = lst[:, None] + np.arange(3)                    # loop indices of each triangle
tri = lv[li]
uv = np.zeros(len(me.loops) * 2, np.float32); me.uv_layers[0].data.foreach_get('uv', uv); uv = uv.reshape(-1, 2)[li]
import bmesh
cap_mat = bpy.data.materials.new('cap'); cap_mat.use_nodes = True
cp = cap_mat.node_tree.nodes['Principled BSDF']; cp.inputs['Base Color'].default_value = (0.012, 0.012, 0.013, 1)
cp.inputs['Roughness'].default_value = 1.0; cp.inputs['Specular IOR Level'].default_value = 0.0   # no sheen: caps read as dark openings

def cap_holes(m):
    # weld the AI model's texture seams, then close every opening left by the cut with a matte black cap
    bm = bmesh.new(); bm.from_mesh(m)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=2e-4)
    edges = [e for e in bm.edges if e.is_boundary]
    new = bmesh.ops.holes_fill(bm, edges=edges, sides=0)['faces']
    if new:
        tri_ = bmesh.ops.triangulate(bm, faces=new)['faces']
        for f in tri_: f.material_index = 1; f.smooth = False
        bmesh.ops.recalc_face_normals(bm, faces=tri_)
    bm.to_mesh(m); bm.free(); m.update()
    return len(new)

objs = {}
for p, name in enumerate(PART_NAMES):
    sel = np.nonzero(fpart == p)[0]
    t = tri[sel]; used, inv = np.unique(t, return_inverse=True)
    m2 = bpy.data.meshes.new(name)
    m2.vertices.add(len(used)); m2.vertices.foreach_set('co', co[used].reshape(-1))       # world coordinates; objects keep identity transforms
    m2.loops.add(len(sel) * 3); m2.loops.foreach_set('vertex_index', inv.reshape(-1).astype(np.int32))
    m2.polygons.add(len(sel)); m2.polygons.foreach_set('loop_start', (np.arange(len(sel)) * 3).astype(np.int32))
    m2.uv_layers.new(); m2.uv_layers[0].data.foreach_set('uv', uv[sel].reshape(-1))
    m2.update(); m2.shade_smooth(); m2.materials.append(mat); m2.materials.append(cap_mat)
    if os.environ.get('CAPS', '0') == '1': cap_holes(m2)
    o = bpy.data.objects.new(name, m2); bpy.context.scene.collection.objects.link(o); objs[name] = o
bpy.data.objects.remove(src)
print('parts', sorted(objs), flush=True)

# the AI texture reads mid-grey; darken it back to the real black mesh/plastic
nt = mat.node_tree; pb = [n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'][0]
if pb.inputs['Base Color'].links:
    lk = pb.inputs['Base Color'].links[0]; g = nt.nodes.new('ShaderNodeGamma'); g.inputs[1].default_value = 2.2
    nt.links.new(lk.from_socket, g.inputs[0]); nt.links.new(g.outputs[0], pb.inputs['Base Color'])
# cap the texture's brightness: the AI texture has light blotches in the mesh fabric that read as patches
sepc = nt.nodes.new('ShaderNodeSeparateColor'); comb = nt.nodes.new('ShaderNodeCombineColor')
src_sock = pb.inputs['Base Color'].links[0].from_socket
nt.links.new(src_sock, sepc.inputs[0])
for ch in range(3):
    mn = nt.nodes.new('ShaderNodeMath'); mn.operation = 'MINIMUM'; mn.inputs[1].default_value = 0.045
    nt.links.new(sepc.outputs[ch], mn.inputs[0]); nt.links.new(mn.outputs[0], comb.inputs[ch])
nt.links.new(comb.outputs[0], pb.inputs['Base Color'])
# black plastic/fabric: no metal, soft low sheen (the ORM map's metallic made it read grey)
chrome_src = mat.copy()
for l in list(pb.inputs['Metallic'].links): nt.links.remove(l)
pb.inputs['Metallic'].default_value = 0.0
pb.inputs['Specular IOR Level'].default_value = 0.25
# roughness floor: glossy spots in the AI roughness map mirrored the bright ceiling as white patches
rl = pb.inputs['Roughness'].links[0].from_socket
rmax = nt.nodes.new('ShaderNodeMath'); rmax.operation = 'MAXIMUM'; rmax.inputs[1].default_value = 0.6
nt.links.new(rl, rmax.inputs[0]); nt.links.new(rmax.outputs[0], pb.inputs['Roughness'])

# polished chrome for the base and gas lift (keeps the texture's dark details)
chrome = chrome_src; chrome.name = 'chrome'
bsdf = [n for n in chrome.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'][0]
for l in list(bsdf.inputs['Metallic'].links) + list(bsdf.inputs['Roughness'].links) + list(bsdf.inputs['Base Color'].links):
    chrome.node_tree.links.remove(l)
bsdf.inputs['Base Color'].default_value = (0.62, 0.62, 0.65, 1)
bsdf.inputs['Metallic'].default_value = 1.0; bsdf.inputs['Roughness'].default_value = 0.14
for n in ('base', 'gas_lift'):
    objs[n].data.materials[0] = chrome

def black_backfaces(m):
    # the inside of a cut opening shows the model's back faces: make them flat black so openings read as dark recesses
    nt = m.node_tree; out = [n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'][0]
    surf = out.inputs['Surface'].links[0].from_socket
    geo = nt.nodes.new('ShaderNodeNewGeometry'); dark = nt.nodes.new('ShaderNodeBsdfDiffuse')
    dark.inputs['Color'].default_value = (0.004, 0.004, 0.004, 1)
    mix = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(geo.outputs['Backfacing'], mix.inputs[0]); nt.links.new(surf, mix.inputs[1]); nt.links.new(dark.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs['Surface'])
for m in (mat, chrome): black_backfaces(m)

def add_glow(m):
    nt = m.node_tree; out = [n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'][0]
    surf = out.inputs['Surface'].links[0].from_socket
    lw = nt.nodes.new('ShaderNodeLayerWeight'); lw.inputs['Blend'].default_value = 0.2
    pw = nt.nodes.new('ShaderNodeMath'); pw.operation = 'POWER'; pw.inputs[1].default_value = 4.0
    at = nt.nodes.new('ShaderNodeAttribute'); at.attribute_type = 'OBJECT'; at.attribute_name = 'glow'
    mu = nt.nodes.new('ShaderNodeMath'); mu.operation = 'MULTIPLY'
    mu2 = nt.nodes.new('ShaderNodeMath'); mu2.operation = 'MULTIPLY'; mu2.inputs[1].default_value = 7.0
    em = nt.nodes.new('ShaderNodeEmission'); em.inputs['Color'].default_value = (0.82, 0.92, 1.0, 1)
    add = nt.nodes.new('ShaderNodeAddShader')
    L = nt.links
    L.new(lw.outputs['Facing'], pw.inputs[0]); L.new(pw.outputs[0], mu.inputs[0]); L.new(at.outputs['Fac'], mu.inputs[1])
    L.new(mu.outputs[0], mu2.inputs[0]); L.new(mu2.outputs[0], em.inputs['Strength'])
    L.new(surf, add.inputs[0]); L.new(em.outputs[0], add.inputs[1]); L.new(add.outputs[0], out.inputs['Surface'])
for m in (mat, chrome, cap_mat): add_glow(m)

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
ramp.color_ramp.elements[0].position = 0.30; ramp.color_ramp.elements[0].color = (0.04, 0.04, 0.045, 1)
ramp.color_ramp.elements[1].position = 0.75; ramp.color_ramp.elements[1].color = (0.95, 0.95, 0.97, 1)
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
    az = math.radians(228 + 22 * math.sin(math.pi * t / 4.8))   # orbit out and back: last frame = first frame
    e = smooth((t - T0) / 1.7) * (1 - smooth((t - HOLD_END) / 1.7))   # pull back gently while the parts are apart
    dist = 3.0 + 1.1 * e; el = math.radians(10)
    target = mathutils.Vector((-0.08 - 0.08 * e, 0.0, 0.0))
    pos = target + dist * mathutils.Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
    camo.location = pos; camo.rotation_euler = (target - pos).to_track_quat('-Z', 'Y').to_euler()

def pose(t):
    for n, (i, off) in MOVES.items():
        k = amount(i, t); objs[n].location = (off[0] * k, off[1] * k, off[2] * k)
        objs[n]['glow'] = 4 * k * (1 - k)                 # flares while the part breaks away and while it locks back in
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
