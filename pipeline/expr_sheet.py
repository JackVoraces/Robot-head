# bpy: build rig, render every expression (face close-up) -> contact sheet
import bpy, runpy, sys, json, os, math, time
from mathutils import Vector
a=sys.argv[sys.argv.index('--')+1:]
OBJ,DATA,OUT=a[:3]; res=int(a[3]) if len(a)>3 else 420; spp=int(a[4]) if len(a)>4 else 24
view=a[5] if len(a)>5 else 'front'
only=a[6].split(',') if len(a)>6 and a[6] else None
bpy.ops.wm.read_factory_settings(use_empty=True)
g=runpy.run_path('/home/user/Robot-head/blender/build_robot_head.py',init_globals={'OBJ_PATH':OBJ,'DATA_DIR':DATA,'BUILD_OPTS':{'demo_animation':False}})
rig=json.load(open(os.path.join(DATA,'rig.json')))
sc=bpy.context.scene; ctrl=bpy.data.objects['CTRL_FACE']
sc.render.engine='CYCLES'; sc.cycles.device='CPU'; sc.cycles.samples=spp
try: sc.cycles.use_denoising=True
except Exception: pass
sc.render.resolution_x=res; sc.render.resolution_y=int(res*1.15)
cam=sc.camera
VIEWS={'front':((0.0,-4.6,0.36),(0,0,0.32),150),'q34':((2.6,-3.8,0.45),(0,0,0.32),150),'q34r':((-2.6,-3.8,0.45),(0,0,0.32),150),'side':((4.6,-0.4,0.36),(0,0,0.32),150)}
loc,tgt,lens=VIEWS[view]; cam.location=Vector(loc); cam.data.lens=lens
cam.rotation_euler=(Vector(tgt)-cam.location).to_track_quat('-Z','Y').to_euler()
names=only or list(rig['expressions'].keys())
outs=[]
for n in names:
    g['apply_expression'](rig,ctrl,n)
    bpy.context.view_layer.update()
    sc.render.filepath=f'{OUT}_{view}_{n}.png'; bpy.ops.render.render(write_still=True); outs.append(sc.render.filepath)
json.dump(outs,open(f'{OUT}_{view}_list.json','w'))
