# usage: bpy-python render_colors.py -- mesh.npz colors.npy out_prefix [views=front,q34,left,back] [res=700]
# colors.npy: (nF,3) float in 0..1 per FACE, or (nV,3) per VERTEX
import bpy, sys, math, mathutils, numpy as np, time
args=sys.argv[sys.argv.index('--')+1:]
MESH, COL, OUT = args[:3]
views=(args[3] if len(args)>3 else 'front,q34,left,back').split(',')
res=int(args[4]) if len(args)>4 else 700
zoom=float(args[5]) if len(args)>5 else 1.0
center_z=float(args[6]) if len(args)>6 else None
bpy.ops.wm.read_factory_settings(use_empty=True)
d=np.load(MESH); V=d['V']; F=d['F']
C=np.load(COL).astype(np.float32)
me=bpy.data.meshes.new('m')
me.vertices.add(len(V)); me.vertices.foreach_set('co',V.astype(np.float32).ravel())
me.loops.add(F.size); me.loops.foreach_set('vertex_index',F.astype(np.int32).ravel())
me.polygons.add(len(F)); me.polygons.foreach_set('loop_start',np.arange(0,F.size,3,dtype=np.int32)); 
me.update(calc_edges=True)
dom='FACE' if len(C)==len(F) else 'POINT'
at=me.color_attributes.new('col','FLOAT_COLOR',dom)
C4=np.concatenate([C,np.ones((len(C),1),np.float32)],1)
at.data.foreach_set('color',C4.ravel())
ob=bpy.data.objects.new('m',me); bpy.context.scene.collection.objects.link(ob)
ob.rotation_euler=(math.pi/2,0,0)  # OBJ Y-up -> Z-up like importer
sc=bpy.context.scene
sc.render.engine='CYCLES'; sc.cycles.device='CPU'; sc.cycles.samples=12; sc.cycles.use_denoising=False
sc.render.resolution_x=int(res*0.78); sc.render.resolution_y=res
sc.view_settings.view_transform='Standard'
w=bpy.data.worlds.new('w'); sc.world=w; w.use_nodes=True
w.node_tree.nodes['Background'].inputs[0].default_value=(1,1,1,1); w.node_tree.nodes['Background'].inputs[1].default_value=0.35
m=bpy.data.materials.new('vc'); m.use_nodes=True
nt=m.node_tree; b=nt.nodes['Principled BSDF']; b.inputs['Roughness'].default_value=0.6
a=nt.nodes.new('ShaderNodeAttribute'); a.attribute_name='col'; nt.links.new(a.outputs['Color'],b.inputs['Base Color'])
ob.data.materials.append(m)
L=bpy.data.lights.new('k','SUN'); L.energy=2.2; L.angle=0.3; lo=bpy.data.objects.new('k',L); sc.collection.objects.link(lo)
cam=bpy.data.cameras.new('c'); cam.type='ORTHO'; cam.ortho_scale=2.05/zoom
co=bpy.data.objects.new('c',cam); sc.collection.objects.link(co); sc.camera=co
ctr=mathutils.Vector((0,0,0 if center_z is None else center_z))
VD={'front':(0,-1,0),'back':(0,1,0),'left':(-1,0,0),'right':(1,0,0),'q34':(0.7,-0.7,0.15),'q34l':(-0.7,-0.7,0.15),'top':(0,-0.25,1),'q34b':(0.7,0.7,0.2),'low':(0,-1,-0.5)}
for name in views:
    dv=mathutils.Vector(VD[name]).normalized()
    co.location=ctr+dv*5
    q=(-dv).to_track_quat('-Z','Y'); co.rotation_euler=q.to_euler()
    # key light from upper-left of camera
    lq=(-(dv+mathutils.Vector((0,0,0.8))+q@mathutils.Vector((-0.6,0,0)))).normalized().to_track_quat('-Z','Y')
    lo.rotation_euler=lq.to_euler()
    sc.render.filepath=f'{OUT}_{name}.png'
    bpy.ops.render.render(write_still=True)
