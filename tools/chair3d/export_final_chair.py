# Package the final rigged chair for delivery: textures down to TEX px (JPEG, packed), then
#   OUT.blend  - the rigged model (CHAIR_RIG controls, drivers)  [zstd-compressed]
#   OUT.glb    - static model at rest, separate named parts, Draco-compressed geometry (for viewers / web / other apps)
# Usage: python export_final_chair.py in.blend OUT   (env: TEX 2048, DRACO 1)
import bpy, sys, os
src, out = sys.argv[-2:]
bpy.ops.wm.open_mainfile(filepath=src)
TEX = int(os.environ.get('TEX', 2048))
for im in bpy.data.images:
    if im.size[0] > TEX:
        im.scale(TEX, TEX * im.size[1] // im.size[0])
        im.file_format = 'JPEG'; im.filepath_raw = '//%s.jpg' % im.name.rsplit('.', 1)[0]
        bpy.context.scene.render.image_settings.quality = 92
        im.pack()
    print('IMG', im.name, im.size[:], flush=True)
bpy.ops.wm.save_as_mainfile(filepath=out + '.blend', compress=True); print('saved', out + '.blend', flush=True)
draco = os.environ.get('DRACO', '1') == '1'
bpy.ops.export_scene.gltf(filepath=out + '.glb', export_format='GLB', export_apply=True, export_image_format='JPEG',
                          export_draco_mesh_compression_enable=draco, export_draco_mesh_compression_level=6,
                          export_draco_position_quantization=14, export_draco_normal_quantization=10,
                          export_draco_texcoord_quantization=12)
print('saved', out + '.glb', flush=True)
