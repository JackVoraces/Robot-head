# bpy script: renders shaded colour + exact ID map + world-position map for a zone.
# args after --: mesh.npz labels.npy colors.npy out_prefix dir_x,dir_y,dir_z zoom cx,cy,cz res
import bpy, sys, math, mathutils, numpy as np
a=sys.argv[sys.argv.index('--')+1:]
MESH,LAB,COL,OUT=a[:4]; dv=mathutils.Vector([float(x) for x in a[4].split(',')]).normalized()
zoom=float(a[5]); ctr=mathutils.Vector([float(x) for x in a[6].split(',')]); res=int(a[7])
bpy.ops.wm.read_factory_settings(use_empty=True)
d=np.load(MESH); V=d['V']; F=d['F']
lab=np.load(LAB).astype(np.float32); C=np.load(COL).astype(np.float32)
me=bpy.data.meshes.new('m')
me.vertices.add(len(V)); me.vertices.foreach_set('co',V.astype(np.float32).ravel())
me.loops.add(F.size); me.loops.foreach_set('vertex_index',F.astype(np.int32).ravel())
me.polygons.add(len(F)); me.polygons.foreach_set('loop_start',np.arange(0,F.size,3,dtype=np.int32))
me.update(calc_edges=True)
at=me.color_attributes.new('col','FLOAT_COLOR','FACE'); at.data.foreach_set('color',np.concatenate([C,np.ones((len(C),1),np.float32)],1).ravel())
li=me.attributes.new('lid','FLOAT','FACE'); li.data.foreach_set('value',lab+1.0)
ob=bpy.data.objects.new('m',me); bpy.context.scene.collection.objects.link(ob)
ob.rotation_euler=(math.pi/2,0,0)
sc=bpy.context.scene
sc.render.engine='CYCLES'; sc.cycles.device='CPU'; sc.cycles.use_denoising=False
sc.render.resolution_x=res; sc.render.resolution_y=res
sc.view_settings.view_transform='Standard'
w=bpy.data.worlds.new('w'); sc.world=w; w.use_nodes=True
bg=w.node_tree.nodes['Background']; bg.inputs[0].default_value=(1,1,1,1)
# materials
ms=bpy.data.materials.new('shade'); ms.use_nodes=True; nt=ms.node_tree; b=nt.nodes['Principled BSDF']; b.inputs['Roughness'].default_value=0.6
at_n=nt.nodes.new('ShaderNodeAttribute'); at_n.attribute_name='col'; nt.links.new(at_n.outputs['Color'],b.inputs['Base Color'])
mi=bpy.data.materials.new('ids'); mi.use_nodes=True; nt=mi.node_tree; nt.nodes.remove(nt.nodes['Principled BSDF'])
em=nt.nodes.new('ShaderNodeEmission'); an=nt.nodes.new('ShaderNodeAttribute'); an.attribute_name='lid'
geo=nt.nodes.new('ShaderNodeNewGeometry'); comb=nt.nodes.new('ShaderNodeCombineXYZ')
nt.links.new(an.outputs['Fac'],comb.inputs[0]); nt.links.new(comb.outputs[0],em.inputs['Color'])
nt.links.new(em.outputs[0],nt.nodes['Material Output'].inputs[0])
mp=bpy.data.materials.new('pos'); mp.use_nodes=True; nt=mp.node_tree; nt.nodes.remove(nt.nodes['Principled BSDF'])
em2=nt.nodes.new('ShaderNodeEmission'); g2=nt.nodes.new('ShaderNodeNewGeometry')
nt.links.new(g2.outputs['Position'],em2.inputs['Color']); nt.links.new(em2.outputs[0],nt.nodes['Material Output'].inputs[0])
ob.data.materials.append(ms)
L=bpy.data.lights.new('k','SUN'); L.energy=2.2; L.angle=0.3; lo=bpy.data.objects.new('k',L); sc.collection.objects.link(lo)
cam=bpy.data.cameras.new('c'); cam.type='ORTHO'; cam.ortho_scale=2.05/zoom; cam.clip_end=100
co=bpy.data.objects.new('c',cam); sc.collection.objects.link(co); sc.camera=co
co.location=ctr+dv*6
q=(-dv).to_track_quat('-Z','Y'); co.rotation_euler=q.to_euler()
lq=(-(dv+mathutils.Vector((0,0,0.8))+q@mathutils.Vector((-0.6,0,0)))).normalized().to_track_quat('-Z','Y'); lo.rotation_euler=lq.to_euler()
# pass 1 shaded
bg.inputs[1].default_value=0.35; sc.cycles.samples=12; sc.render.film_transparent=False
sc.render.image_settings.file_format='PNG'
sc.render.filepath=OUT+'_shade.png'; bpy.ops.render.render(write_still=True)
# pass 2/3 exact
bg.inputs[1].default_value=0.0; sc.cycles.samples=1; sc.cycles.filter_width=0.01; sc.render.film_transparent=True
lo.hide_render=True
sc.render.image_settings.file_format='OPEN_EXR'; sc.render.image_settings.color_depth='32'
for nm,mat in (('ids',mi),('pos',mp)):
    ob.data.materials[0]=mat
    sc.render.filepath=OUT+f'_{nm}.exr'; bpy.ops.render.render(write_still=True)
    im=bpy.data.images.load(OUT+f'_{nm}.exr'); px=np.empty(len(im.pixels),np.float32); im.pixels.foreach_get(px)
    px=px.reshape(im.size[1],im.size[0],4)[::-1]  # top row first
    np.save(OUT+f'_{nm}.npy',px)
