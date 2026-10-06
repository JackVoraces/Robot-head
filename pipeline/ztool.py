#!/usr/bin/env python3
"""Zone inspection toolkit.
  ztool.py render LABELS.npy ZONE OUTDIR [--res N]   -> OUTDIR/ZONE_shade.png, _annot.png, _ids.npy, _pos.npy
  ztool.py pick OUTDIR/ZONE x y                       -> region id + world xyz at pixel (x,y) of the annotated image
  ztool.py info LABELS.npy id [id...]                 -> region stats + neighbours
  ztool.py highlight LABELS.npy ZONE OUTDIR id[,id..] -> render with only those regions coloured (others grey)
  ztool.py point OUTDIR/ZONE id                       -> world xyz at region id's label position in that zone render
Zones: see ZONES dict. World coords are Blender Z-up; character faces -Y; character LEFT is +X.
"""
import sys, os, json, subprocess, numpy as np
W=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,W)
import seglib
BPY=seglib.BPY
ZONES={
 'face':      dict(dir=(0,-1,0.12), zoom=2.3, ctr=(0,0,0.28)),
 'eyes':      dict(dir=(0,-1,0.05), zoom=4.0, ctr=(0,0,0.33)),
 'muzzle':    dict(dir=(0,-1,-0.25),zoom=2.3, ctr=(0,0,0.02)),
 'crown':     dict(dir=(0,-0.45,1), zoom=2.0, ctr=(0,0,0.6)),
 'side_L':    dict(dir=(1,-0.15,0.05),zoom=2.0, ctr=(0,0,0.32)),
 'side_R':    dict(dir=(-1,-0.15,0.05),zoom=2.0, ctr=(0,0,0.32)),
 'q34_L':     dict(dir=(0.7,-0.7,0.1),zoom=2.0, ctr=(0,0,0.3)),
 'q34_R':     dict(dir=(-0.7,-0.7,0.1),zoom=2.0, ctr=(0,0,0.3)),
 'back':      dict(dir=(0,1,0.12), zoom=2.0, ctr=(0,0,0.38)),
 'neck_front':dict(dir=(0,-1,-0.05),zoom=2.3, ctr=(0,0,-0.3)),
 'neck_L':    dict(dir=(0.8,-0.6,0),zoom=2.3, ctr=(0,0,-0.25)),
 'neck_R':    dict(dir=(-0.8,-0.6,0),zoom=2.3, ctr=(0,0,-0.25)),
 'neck_back': dict(dir=(0,1,-0.05), zoom=2.3, ctr=(0,0,-0.15)),
 'bust_front':dict(dir=(0,-1,0.1), zoom=1.0, ctr=(0,0,0)),
 'bust_back': dict(dir=(0,1,0.1),  zoom=1.0, ctr=(0,0,0)),
 'bust_q34':  dict(dir=(0.7,-0.7,0.2),zoom=1.0, ctr=(0,0,0)),
}
def _edges():
    return np.load(os.path.join(W,'edges.npz'))
def render(labels_path,zone,outdir,res=1000,colors=None,tag=None):
    os.makedirs(outdir,exist_ok=True)
    lab=np.load(labels_path)
    z=ZONES[zone]; out=os.path.abspath(os.path.join(outdir,tag or zone))
    if colors is None: colors=seglib.distinct_colors(lab)
    cp=out+'_cols.npy'; np.save(cp,colors.astype(np.float32))
    cmd=[BPY,os.path.join(W,'tools','zrender_bpy.py'),'--',os.path.join(W,'mesh.npz'),os.path.abspath(labels_path),cp,out,
         ','.join(map(str,z['dir'])),str(z['zoom']),','.join(map(str,z['ctr'])),str(res)]
    r=subprocess.run(cmd,capture_output=True,text=True)
    if r.returncode: print(r.stdout[-3000:],r.stderr[-3000:]); raise SystemExit(1)
    annotate(out)
    os.remove(cp)
    print(out+'_annot.png')
def annotate(out,min_px=60):
    from PIL import Image, ImageDraw, ImageFont
    from scipy import ndimage
    ids=np.load(out+'_ids.npy'); idm=np.rint(ids[...,0]).astype(np.int64)-1; alpha=ids[...,3]
    idm[alpha<0.5]=-1
    np.save(out+'_idmap.npy',idm.astype(np.int32))
    img=Image.open(out+'_shade.png').convert('RGB'); dr=ImageDraw.Draw(img)
    H,Wd=idm.shape
    fs=max(11,int(H/70))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',fs)
    # boundaries in black
    bnd=np.zeros_like(idm,bool); bnd[:-1]|=idm[:-1]!=idm[1:]; bnd[:,:-1]|=idm[:,:-1]!=idm[:,1:]
    arr=np.array(img); arr[bnd]=(arr[bnd]*0.25).astype(np.uint8); img=Image.fromarray(arr); dr=ImageDraw.Draw(img)
    vis={}
    for l in np.unique(idm[idm>=0]):
        m=idm==l; n=int(m.sum())
        if n<min_px: continue
        sl=ndimage.find_objects(m.astype(np.int8))[0]
        sub=m[sl]; dt=ndimage.distance_transform_edt(np.pad(sub,1))[1:-1,1:-1]
        y,x=np.unravel_index(np.argmax(dt),dt.shape); y+=sl[0].start; x+=sl[1].start
        vis[int(l)]=dict(px=n,xy=[int(x),int(y)])
        t=str(l); tw=dr.textlength(t,font=font)
        dr.rectangle([x-tw/2-2,y-fs/2-1,x+tw/2+2,y+fs/2+2],fill=(255,255,255))
        dr.text((x-tw/2,y-fs/2),t,fill=(0,0,0),font=font)
    img.save(out+'_annot.png')
    json.dump(vis,open(out+'_visible.json','w'))
def pick(prefix,x,y):
    idm=np.load(prefix+'_idmap.npy'); pos=np.load(prefix+'_pos.npy')
    return dict(id=int(idm[y,x]),xyz=[round(float(v),5) for v in pos[y,x,:3]])
def to_world(c):  # OBJ (y-up) -> Blender world (z-up)
    c=np.asarray(c); return np.stack([c[...,0],-c[...,2],c[...,1]],-1)
def info(labels_path,ids):
    lab=np.load(labels_path); E=_edges(); fa,fb,elen,ar,cen=E['fa'],E['fb'],E['elen'],E['ar'],E['cen']
    Bz=np.load(os.path.join(W,'barrier.npz')); eb=Bz['eb_max']
    out=[]
    for r in ids:
        m=lab==r; A=ar[m].sum(); c=to_world((cen[m]*ar[m,None]).sum(0)/A)
        wc=to_world(cen[m]); bb=[wc.min(0).round(4).tolist(),wc.max(0).round(4).tolist()]
        e=(lab[fa]==r)^(lab[fb]==r); other=np.where(lab[fa][e]==r,lab[fb][e],lab[fa][e])
        nb={}
        for o,l,b in zip(other,elen[e],eb[e]):
            d=nb.setdefault(int(o),[0.0,0.0]); d[0]+=l; d[1]+=l*(b>20)
        nbs=sorted([(k,round(v[0],4),round(v[1]/v[0],2)) for k,v in nb.items()],key=lambda t:-t[1])
        out.append(dict(id=int(r),faces=int(m.sum()),area=round(float(A),6),centroid=c.round(4).tolist(),bbox=bb,
                        neighbours_id_len_creasefrac=nbs[:15]))
    return out
if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='render':
        res=1000
        if '--res' in sys.argv: res=int(sys.argv[sys.argv.index('--res')+1])
        render(sys.argv[2],sys.argv[3],sys.argv[4],res=res)
    elif cmd=='pick':
        print(json.dumps(pick(sys.argv[2],int(sys.argv[3]),int(sys.argv[4]))))
    elif cmd=='info':
        print(json.dumps(info(sys.argv[2],[int(x) for x in sys.argv[3:]]),indent=1))
    elif cmd=='point':
        # world xyz at the label position of region id in a rendered zone: ztool.py point PREFIX id
        vis=json.load(open(sys.argv[2]+'_visible.json')); x,y=vis[sys.argv[3]]['xy']
        print(json.dumps(pick(sys.argv[2],x,y)))
    elif cmd=='highlight':
        lab=np.load(sys.argv[2]); ids=[int(x) for x in sys.argv[5].split(',')]
        c=np.full((len(lab),3),0.55); 
        for k,i in enumerate(ids): c[lab==i]=seglib.PAL[k%len(seglib.PAL)]
        render(sys.argv[2],sys.argv[3],sys.argv[4],colors=c,tag=sys.argv[3]+'_hl')
