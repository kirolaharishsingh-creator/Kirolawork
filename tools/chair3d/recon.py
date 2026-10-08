# Orthographic recon renders of the Tripo chair with a coordinate grid, to place part cut boundaries.
import bpy, math, mathutils
from PIL import Image, ImageDraw
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath='chair.glb')
sc = bpy.context.scene; sc.render.engine = 'CYCLES'; sc.cycles.samples = 8
sc.render.resolution_x = sc.render.resolution_y = 900
w = bpy.data.worlds.new('w'); sc.world = w; w.use_nodes = True
w.node_tree.nodes['Background'].inputs[0].default_value = (1, 1, 1, 1); w.node_tree.nodes['Background'].inputs[1].default_value = 2.5
cam = bpy.data.cameras.new('c'); cam.type = 'ORTHO'; cam.ortho_scale = 1.1
co = bpy.data.objects.new('c', cam); sc.collection.objects.link(co); sc.camera = co
S = 1.1
# view name, camera position, screen-right axis, screen-up axis
views = [('side', (0, -3, 0), (1, 0, 0), (0, 0, 1)),     # looking +Y: x to the right
         ('front', (3, 0, 0), (0, 1, 0), (0, 0, 1)),     # looking -X: y to the right
         ('top', (0, 0, 3), (1, 0, 0), (0, 1, 0))]
for name, pos, rx, ux in views:
    co.location = pos
    d = -mathutils.Vector(pos); co.rotation_euler = d.to_track_quat('-Z', 'Y' if name != 'top' else 'Y').to_euler()
    if name == 'top': co.rotation_euler = (0, 0, 0)
    sc.render.filepath = f'/home/user/{name}.png'; bpy.ops.render.render(write_still=True)
    im = Image.open(f'/home/user/{name}.png').convert('RGB'); dr = ImageDraw.Draw(im)
    for i in range(-11, 12):
        v = i * 0.05; p = int(450 + v / S * 900)
        col = (255, 0, 0) if i % 2 == 0 else (255, 170, 170)
        dr.line([(p, 0), (p, 899)], fill=col, width=1); dr.line([(0, 900 - p), (899, 900 - p)], fill=col, width=1)
        if i % 2 == 0:
            dr.text((p + 2, 2), f'{v:.1f}', fill=(200, 0, 0)); dr.text((2, 900 - p - 12), f'{v:.1f}', fill=(200, 0, 0))
    im.convert('RGB').save(f'/home/user/{name}.jpg', quality=70)
    print('done', name, flush=True)
