# Load chair.glb once and cache world-space vertices, triangles and loose-shell labels as .npy.
import bpy, numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath='chair.glb')
o = [o for o in bpy.context.scene.objects if o.type == 'MESH'][0]; me = o.data
nv = len(me.vertices); co = np.zeros(nv * 3); me.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
M = np.array(o.matrix_world); co = co @ M[:3, :3].T + M[:3, 3]
nf = len(me.polygons); ls = np.zeros(nf, int); me.polygons.foreach_get('loop_start', ls)
lv = np.zeros(len(me.loops), int); me.loops.foreach_get('vertex_index', lv)
tri = lv[ls[:, None] + np.arange(3)]
e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]]])
n, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nv, nv)), directed=False)
np.save('co.npy', co.astype(np.float32)); np.save('tri.npy', tri.astype(np.int32)); np.save('lab.npy', lab)
print('prep', nv, len(tri), n)
