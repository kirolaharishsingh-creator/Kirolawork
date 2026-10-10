import bpy, sys, math, mathutils, os
f=sys.argv[-2]; out=sys.argv[-1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=f)
ms=[o for o in bpy.context.scene.objects if o.type=='MESH']
nv=sum(len(o.data.vertices) for o in ms); nf=sum(len(o.data.polygons) for o in ms)
pts=[o.matrix_world@v.co for o in ms for v in list(o.data.vertices)[::50]]
lo=[min(p[i] for p in pts) for i in range(3)];hi=[max(p[i] for p in pts) for i in range(3)]
print('MESHES',len(ms),'verts',nv,'faces',nf,'bbox',[round(h-l,3) for l,h in zip(lo,hi)])
print('MATS',[m.name for o in ms for m in o.data.materials],[ (i.name,i.size[:]) for i in bpy.data.images])
sc=bpy.context.scene; sc.render.engine='CYCLES'; sc.cycles.samples=24; sc.render.resolution_x=sc.render.resolution_y=900
sc.render.film_transparent=False
w=bpy.data.worlds.new('w'); sc.world=w; w.use_nodes=True; w.node_tree.nodes['Background'].inputs[0].default_value=(0.85,0.85,0.85,1); w.node_tree.nodes['Background'].inputs[1].default_value=1.0
c=mathutils.Vector([(l+h)/2 for l,h in zip(lo,hi)]); r=max(h-l for l,h in zip(lo,hi))
for nm,loc,e in (('k',(2,-2,3),600),('f',(-2,-1,1),250)):
    L=bpy.data.lights.new(nm,'AREA');L.energy=e*r*r;L.size=2*r;o=bpy.data.objects.new(nm,L);sc.collection.objects.link(o);o.location=c+mathutils.Vector(loc)*r;o.rotation_euler=(c-o.location).to_track_quat('-Z','Y').to_euler()
cam=bpy.data.cameras.new('c');co=bpy.data.objects.new('c',cam);sc.collection.objects.link(co);sc.camera=co
import os
for i,(az,el) in enumerate(((35,25),(215,25),(90,5),(0,80))):
    a,b=math.radians(az),math.radians(el);co.location=c+2.2*r*mathutils.Vector((math.cos(b)*math.cos(a),math.cos(b)*math.sin(a),math.sin(b)))
    co.rotation_euler=(c-co.location).to_track_quat('-Z','Y').to_euler()
    sc.render.filepath=f'{out}_{i}.png';bpy.ops.render.render(write_still=True)
