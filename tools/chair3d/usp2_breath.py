# USP shot 2, mesh breathability: the backrest alone, floating at an angle on the light-grey studio backdrop, with
# soft light-blue airflow streaks (no glow) passing through the mesh from behind to the front. Loops seamlessly.
# Usage: python3 usp2_breath.py test|full   (env: BLEND chair_m8c.blend, RES, SAMPLES, DURATION, FRAMES, N streaks,
#        SEE mesh see-through 0..1, AIR_RGB, AIR_ALPHA, SEED)
import bpy, sys, os, math, numpy as np, mathutils

MODE = sys.argv[-1]
FPS = 24; DUR = float(os.environ.get('DURATION', 3)); N = int(round(DUR * FPS))     # frames 0..N-1, frame N == frame 0
W = int(os.environ.get('RES', 1920)); H = W * 9 // 16
bpy.ops.wm.open_mainfile(filepath=os.environ['BLEND'])
sc = bpy.context.scene
for o in list(sc.objects):
    if o.type in ('MESH', 'CURVE') and o.name != 'backrest': bpy.data.objects.remove(o)
br = bpy.data.objects['backrest']
if os.environ.get('PANEL_JSON'):
    # clean rebuild of the backrest for this close-up (Tripo's rim is torn at the top): the backrest's own outline
    # (traced from the model, symmetrised) and curvature, a black rounded frame tube along it and the real mesh
    # fabric inside, with the same materials as the chair
    import json
    J = json.load(open(os.environ['PANEL_JSON'])); O = np.array(J['outline']); cf = np.array(J['surf'])
    # smooth the traced outline (circular Gaussian along it) so the frame is an even curve, not lumpy
    sg = float(os.environ.get('OUT_SMOOTH', 5)); k_ = np.arange(-15, 16); g_ = np.exp(-0.5 * (k_ / sg) ** 2); g_ /= g_.sum()
    O = np.stack([np.convolve(np.r_[O[-15:, i], O[:, i], O[:15, i]], g_, 'same')[15:-15] for i in range(2)], 1)
    surf = lambda x, z: cf[0] + cf[1] * x + cf[2] * z + cf[3] * x * x + cf[4] * x * z + cf[5] * z * z
    mats = {m.name: m for m in br.data.materials if m}
    plastic = next(m for n, m in mats.items() if 'plastic' in n); fabric = next(m for n, m in mats.items() if n.startswith('mesh_fabric'))
    def offset(P, d):                                   # move a closed outline inward by d (positive = inward)
        nrm = np.roll(P, -1, 0) - np.roll(P, 1, 0); nrm = np.c_[-nrm[:, 1], nrm[:, 0]]; nrm /= np.linalg.norm(nrm, axis=1)[:, None]
        cen = P.mean(0); sgn = np.sign(((cen - P) * nrm).sum(1).mean()); return P + sgn * d * nrm
    def inside(P, pts):                                 # even-odd rule
        x, y = pts[:, 0][:, None], pts[:, 1][:, None]; x1, y1 = P[:, 0][None], P[:, 1][None]; x2, y2 = np.roll(P[:, 0], -1)[None], np.roll(P[:, 1], -1)[None]
        c = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1); return c.sum(1) % 2 == 1
    RIM = float(os.environ.get('RIM_W', 0.024))
    inner = offset(O, RIM - 0.006)                      # the mesh runs a little under the frame
    st = 0.002; gx = np.arange(O[:, 0].min(), O[:, 0].max() + st, st); gz = np.arange(O[:, 1].min(), O[:, 1].max() + st, st)
    GX, GZ = np.meshgrid(gx, gz); ins = inside(inner, np.c_[GX.ravel(), GZ.ravel()]).reshape(GX.shape)
    import bmesh as _bm
    bm = _bm.new(); vid = {}
    for i in range(GX.shape[0]):
        for j in range(GX.shape[1]):
            if ins[i, j]: vid[i, j] = bm.verts.new((GX[i, j], surf(GX[i, j], GZ[i, j]), GZ[i, j]))
    for (i, j), a in list(vid.items()):
        q = [(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)]
        if all(k in vid for k in q): bm.faces.new([vid[k] for k in q])
    me = bpy.data.meshes.new('mesh_panel'); bm.to_mesh(me); bm.free()
    mp = bpy.data.objects.new('mesh_panel', me); sc.collection.objects.link(mp); me.materials.append(fabric)
    vv = np.array([x.co for x in me.vertices]); lv = np.zeros(len(me.loops), int); me.loops.foreach_get('vertex_index', lv); pp = vv[lv]
    me.uv_layers.new(name='mesh_uv').data.foreach_set('uv', np.c_[-pp[:, 0] / 0.194, pp[:, 2] / 0.176].astype(np.float32).ravel())
    me.polygons.foreach_set('use_smooth', np.ones(len(me.polygons), bool))
    sd = mp.modifiers.new('sd', 'SOLIDIFY'); sd.thickness = 0.002; sd.offset = 0
    mid = offset(O, RIM / 2); cu = bpy.data.curves.new('frame_tube', 'CURVE'); cu.dimensions = '3D'
    cu.bevel_depth = RIM / 2; cu.bevel_resolution = 6; sp = cu.splines.new('POLY'); sp.points.add(len(mid) - 1); sp.use_cyclic_u = True
    for pnt, (x, z) in zip(sp.points, mid): pnt.co = (x, surf(x, z) + float(os.environ.get('RIM_BACK', 0.004)), z, 1)
    fo = bpy.data.objects.new('frame_tube', cu); sc.collection.objects.link(fo); cu.materials.append(plastic)
    sm = fo.modifiers.new('sm', 'SUBSURF'); sm.levels = 1; sm.render_levels = 2
    # a moulded rim, not a round tube: flatten the frame's depth about the panel surface
    import bmesh as _b2
    dg = bpy.context.evaluated_depsgraph_get(); fm = bpy.data.meshes.new_from_object(fo.evaluated_get(dg)); fo2 = bpy.data.objects.new('frame', fm)
    sc.collection.objects.link(fo2); bpy.data.objects.remove(fo)
    fl = float(os.environ.get('RIM_FLAT', 0.5))
    for vx in fm.vertices: ys = surf(vx.co.x, vx.co.z); vx.co.y = ys + (vx.co.y - ys) * fl
    fm.polygons.foreach_set('use_smooth', np.ones(len(fm.polygons), bool)); fm.update()
    # the frame in the photos is a deep matte black (fabric-wrapped), not the grey sheen of plain plastic
    rm = bpy.data.materials.new('rim_black'); rm.use_nodes = True; rp = rm.node_tree.nodes['Principled BSDF']
    rp.inputs['Base Color'].default_value = (0.0015, 0.0015, 0.0018, 1); rp.inputs['Roughness'].default_value = float(os.environ.get('RIM_ROUGH', 0.6))
    rp.inputs['Specular IOR Level'].default_value = 0.08
    fm.materials.clear(); fm.materials.append(rm)
    if os.environ.get('YFRAME', '0') == '1':
        # the black Y-frame seen through the mesh in the photos: two straps from the top corners converging at the
        # bottom centre, plus the tongue at the top centre. Drawn from the front photo (15.jpg, backrest box
        # x 345..768, y 505..870 px) as an alpha mask on a black sheet just behind the mesh.
        from PIL import Image as _I, ImageDraw as _D
        R = 2048; im = _I.new('L', (R, R), 0); dr = _D.Draw(im)
        px = lambda pts: [((x - 345) / 423 * R, (y - 505) / 365 * R) for x, y in pts]
        L = [(362, 538), (428, 538), (556, 842), (556, 880), (498, 880)]
        dr.polygon(px(L), 255); dr.polygon(px([(2 * 556 - x, y) for x, y in L]), 255)
        dr.polygon(px([(500, 500), (612, 500), (580, 560), (532, 560)]), 255)
        dr.polygon(px([(534, 500), (578, 500), (578, 596), (534, 596)]), 255)
        (a, b), (c, d) = px([(534, 574), (578, 618)]); dr.ellipse([a, b, c, d], 255)
        im = im.resize((1024, 1024), _I.LANCZOS); im = _I.eval(im, lambda q: 255 - q) if os.environ.get('YF_INV', '1') == '1' else im; mpth = os.path.join(os.getcwd(), 'yframe_mask.png'); im.save(mpth)
        me2 = me.copy(); yf = bpy.data.objects.new('yframe', me2); sc.collection.objects.link(yf)
        yf.location.y = float(os.environ.get('YF_BACK', 0.011)); me2.materials.clear()
        x0, x1, z0, z1 = O[:, 0].min(), O[:, 0].max(), O[:, 1].min(), O[:, 1].max()
        me2.uv_layers.remove(me2.uv_layers['mesh_uv']) if 'mesh_uv' in me2.uv_layers else None
        me2.uv_layers.new(name='yf_uv').data.foreach_set('uv', np.c_[(pp[:, 0] - x0) / (x1 - x0), (pp[:, 2] - z0) / (z1 - z0)].astype(np.float32).ravel())
        ym = bpy.data.materials.new('yframe'); ym.use_nodes = True; yn = ym.node_tree; yp = yn.nodes['Principled BSDF']
        yp.inputs['Base Color'].default_value = (0.012, 0.012, 0.013, 1); yp.inputs['Roughness'].default_value = 0.5
        yp.inputs['Specular IOR Level'].default_value = 0.3
        tx = yn.nodes.new('ShaderNodeTexImage'); tx.image = bpy.data.images.load(mpth); tx.image.colorspace_settings.name = 'Non-Color'
        uvn = yn.nodes.new('ShaderNodeUVMap'); uvn.uv_map = 'yf_uv'; yn.links.new(uvn.outputs[0], tx.inputs[0])
        yn.links.new(tx.outputs['Color'], yp.inputs['Alpha']); me2.materials.append(ym)
        yf.modifiers.new('sd', 'SOLIDIFY').thickness = 0.008
    bpy.data.objects.remove(br); br = mp
    print('clean panel built:', len(me.polygons), 'mesh faces, frame tube', len(mid), 'pts')
v = np.array([br.matrix_world @ x.co for x in br.data.vertices]); lo, hi = v.min(0), v.max(0); C = (lo + hi) / 2
if os.environ.get('PANEL_JSON'): o_ = np.array(J['outline']); lo[0], hi[0], lo[2], hi[2] = o_[:, 0].min(), o_[:, 0].max(), o_[:, 1].min(), o_[:, 1].max(); C = (lo + hi) / 2
# the mesh panel: a little more see-through than on the full chair, so the air visibly passes through it
see = float(os.environ.get('SEE', 0.25))
for m in br.data.materials:
    if m and m.name.startswith('mesh_fabric'):
        for n in m.node_tree.nodes:
            if n.type == 'MIX_SHADER': n.inputs[0].default_value = see

# ---- airflow streaks: thin tubes along smooth paths through the panel; a short segment travels along each path
rng = np.random.default_rng(int(os.environ.get('SEED', 3)))
air = bpy.data.materials.new('air'); air.use_nodes = True; nt = air.node_tree; pb = nt.nodes['Principled BSDF']
rgb = [float(x) for x in os.environ.get('AIR_RGB', '0.55,0.74,1.0').split(',')]
pb.inputs['Base Color'].default_value = (*rgb, 1); pb.inputs['Roughness'].default_value = 0.6
pb.inputs['Alpha'].default_value = float(os.environ.get('AIR_ALPHA', 0.55))
pb.inputs['Specular IOR Level'].default_value = 0.1
streaks = []
NS = int(os.environ.get('N', 24)); SEG = 18
# taper: thin -> full -> thin along every streak, so each one reads as a wisp of air, not a rod
tc = bpy.data.curves.new('taper', 'CURVE'); ts = tc.splines.new('POLY'); tpts = [(0, 0), (0.2, 0.7), (0.45, 1), (0.75, 0.6), (1, 0)]
ts.points.add(len(tpts) - 1)
for p, (x, y) in zip(ts.points, tpts): p.co = (x, y, 0, 1)
taper = bpy.data.objects.new('taper', tc); sc.collection.objects.link(taper); taper.hide_render = True
# per-streak strength: the material's alpha is scaled by the object's colour (alpha channel)
oi = nt.nodes.new('ShaderNodeObjectInfo'); mul = nt.nodes.new('ShaderNodeMath'); mul.operation = 'MULTIPLY'
mul.inputs[1].default_value = float(os.environ.get('AIR_ALPHA', 0.55)); nt.links.new(oi.outputs['Alpha'], mul.inputs[0]); nt.links.new(mul.outputs[0], pb.inputs['Alpha'])
def catmull(pts, n=240):
    P = np.array(pts, float); P = np.vstack([P[0], P, P[-1]]); out = []
    for i in range(1, len(P) - 2):
        for t in np.linspace(0, 1, n // (len(P) - 3), endpoint=False):
            p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-2]); out = np.array(out)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(out, axis=0), axis=1))]; return out, d / d[-1]
for k in range(NS):
    x = rng.uniform(lo[0] + 0.04, hi[0] - 0.04); z = rng.uniform(lo[2] + 0.05, hi[2] - 0.04)
    yb = C[1] + 0.02                                                   # roughly where the panel sits (front is -y)
    lift = rng.uniform(-0.02, 0.06); sway = rng.uniform(-0.05, 0.05)
    pts = [(x + sway * 2 + 0.16, yb + 0.95, z - lift - 0.06), (x + sway + 0.07, yb + 0.45, z - lift * 0.5),
           (x, yb, z), (x - sway - 0.06, yb - 0.45, z + lift * 0.6), (x - sway * 2 - 0.14, yb - 0.95, z + lift + 0.06)]
    path, tt = catmull(pts)
    cu = bpy.data.curves.new(f'air{k}', 'CURVE'); cu.dimensions = '3D'; cu.bevel_depth = float(os.environ.get('THICK', 0.0016)) * rng.uniform(0.7, 1.3)
    cu.bevel_resolution = 2; cu.taper_object = taper; cu.use_fill_caps = False
    sp = cu.splines.new('POLY'); sp.points.add(SEG - 1)
    ob = bpy.data.objects.new(f'air{k}', cu); sc.collection.objects.link(ob); cu.materials.append(air)
    ob.color = (1, 1, 1, rng.uniform(0.45, 1.0))
    streaks.append((ob, rng.uniform(0, 1), rng.uniform(0.22, 0.38), int(rng.integers(1, 3)), path, tt))

def pose(f):
    u = f / N
    for ob, ph, L, laps, path, tt in streaks:
        s = (u * laps + ph) % 1.0                         # integer laps per loop: frame N == frame 0
        a = -L + s * (1 + L)
        t0, t1 = max(a, 0), min(a + L, 1)
        if t1 - t0 < 0.02: ob.hide_render = True; continue
        ob.hide_render = False
        ts_ = np.linspace(t0, t1, SEG); q = np.stack([np.interp(ts_, tt, path[:, i]) for i in range(3)], 1)
        for p, c in zip(ob.data.splines[0].points, q): p.co = (*c, 1)

# ---- scene
sc.render.engine = 'CYCLES'; sc.cycles.device = 'CPU'; sc.cycles.samples = int(os.environ.get('SAMPLES', 32)); sc.cycles.use_denoising = True
sc.render.film_transparent = True; sc.render.resolution_x, sc.render.resolution_y = W, H; sc.render.fps = FPS
sc.view_settings.view_transform = 'Standard'
world = bpy.data.worlds.new('w'); sc.world = world; world.use_nodes = True
world.node_tree.nodes['Background'].inputs[0].default_value = (0.62, 0.62, 0.64, 1); world.node_tree.nodes['Background'].inputs[1].default_value = 0.12
def area(name, loc, energy, size):
    L = bpy.data.lights.new(name, 'AREA'); L.energy = energy; L.size = size
    o = bpy.data.objects.new(name, L); sc.collection.objects.link(o); o.location = mathutils.Vector(C) + mathutils.Vector(loc)
    o.rotation_euler = (mathutils.Vector(C) - o.location).to_track_quat('-Z', 'Y').to_euler()
area('key', (-1.2, -1.4, 1.2), float(os.environ.get('KEY', 60)), 1.6)
area('back', (0.6, 1.6, 0.5), float(os.environ.get('BACK', 45)), 1.2)      # behind the panel: shows the open weave
area('fill', (1.4, -0.8, 0.2), 18, 2.0)
cam = bpy.data.cameras.new('cam'); cam.lens = float(os.environ.get('LENS', 55))
camo = bpy.data.objects.new('cam', cam); sc.collection.objects.link(camo); sc.camera = camo
AZ, EL, D = (float(os.environ.get(k, d)) for k, d in (('CAM_AZ', -55), ('CAM_EL', 16), ('DIST', 1.15)))
def camera(f):
    a = math.radians(AZ + 3 * math.sin(2 * math.pi * f / N)); e = math.radians(EL)          # a gentle drift that loops
    pos = mathutils.Vector(C) + D * mathutils.Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))
    camo.location = pos; camo.rotation_euler = (mathutils.Vector(C) - pos).to_track_quat('-Z', 'Y').to_euler()

frames = [int(x) for x in os.environ.get('FRAMES', '0,36').split(',')] if MODE == 'test' else range(N)
os.makedirs('frames', exist_ok=True)
for f in frames:
    pose(f); camera(f); bpy.context.view_layer.update()
    sc.render.filepath = f'{os.getcwd()}/frames/{f:04d}.png'; bpy.ops.render.render(write_still=True); print('frame', f, flush=True)
