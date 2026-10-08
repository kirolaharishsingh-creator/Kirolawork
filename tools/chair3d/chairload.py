# Import the chair model and bring it into the working frame: front facing +x, z centred on 0 (height ~0.98 m).
# CHAIR_ROT90=1 is for the second model, which faces -y and stands on z = 0.
import os, bpy, mathutils

def load_chair():
    bpy.ops.import_scene.gltf(filepath=os.environ.get('CHAIR', 'chair.glb'))
    o = [o for o in bpy.context.scene.objects if o.type == 'MESH'][0]
    if os.environ.get('CHAIR_ROT90') == '1':
        me = o.data; me.transform(o.matrix_world); o.matrix_world = mathutils.Matrix.Identity(4)
        me.transform(mathutils.Matrix.Rotation(1.5707963267948966, 4, 'Z'))
        zs = [v.co.z for v in me.vertices]; me.transform(mathutils.Matrix.Translation((0, 0, -(min(zs) + max(zs)) / 2)))
        me.update()
    return o
