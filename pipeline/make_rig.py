"""design JSON + base labels  ->  final labels (with cuts), regions.json, rig.json, core  (Blender build data)
usage: make_rig.py BASE_LABELS.npy DESIGN.json OUTDIR [--clothing CLOTHING_LABELS_JSON]"""
import numpy as np, json, lzma, io, sys, os, copy
W=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,os.path.join(W,'tools'))
from relief import relief
def to_world(a): a=np.asarray(a,float); return np.stack([a[...,0],-a[...,2],a[...,1]],-1)
def nrm(v): return v/(np.linalg.norm(v)+1e-12)
d=np.load(os.path.join(W,'mesh.npz')); V,F=d['V'],d['F']
E=np.load(os.path.join(W,'edges.npz')); fa,fb,ev,elen,ar,fn,cen=E['fa'],E['fb'],E['ev'],E['elen'],E['ar'],E['fn'],E['cen']
Vw=to_world(V); FNw=to_world(fn); Cw=to_world(cen)

def frame_for(faces,side):
    Z=nrm((FNw[faces]*ar[faces,None]).sum(0))
    up=np.array([0,0,1.0])
    if abs(Z@up)>0.92: up=np.array([0,-1.0,0])
    Y=nrm(up-(up@Z)*Z); X=np.cross(Y,Z)
    return np.stack([X,Y,Z])          # rows: axes in world
def pivot_for(faces,frame,rule,side):
    vi=np.unique(F[faces]); P=Vw[vi]; c=P.mean(0); L=(P-c)@frame.T
    if isinstance(rule,list): return np.array(rule,float)
    lat=1.0 if side!='R' else -1.0   # X axis points lateral on L, medial on R (front plates)
    def slice_mean(coord,sign,q=0.06):
        x=coord*sign; t=np.quantile(x,1-q); return P[x>=t].mean(0)
    if rule=='lateral_end': return slice_mean(L[:,0],lat)
    if rule=='medial_end':  return slice_mean(L[:,0],-lat)
    if rule=='top_edge':    return slice_mean(L[:,1],1)
    if rule=='bottom_edge': return slice_mean(L[:,1],-1)
    # center: boundary-loop centroid
    fs=set(faces.tolist())
    return c
def main(base,design,out,clothing=None):
    lab=np.load(base).astype(np.int64); des=json.load(open(design))
    nxt=lab.max()+1; cutmap={}
    # ---- planar cuts (brows)
    for cdef in des.get('cuts',[]):
        r=cdef['region']; faces=np.flatnonzero(lab==r)
        P=Cw[faces]; c=(P*ar[faces,None]).sum(0)/ar[faces].sum()
        Z=nrm((FNw[faces]*ar[faces,None]).sum(0))
        Q=P-c; Q=Q-np.outer(Q@Z,Z)
        u,s,vt=np.linalg.svd(Q*np.sqrt(ar[faces,None]),full_matrices=False); e1=vt[0]
        if (c+e1)[0]**2 < (c-e1)[0]**2:  # orient e1 lateral (away from midline)
            pass
        if abs(c[0]+e1[0]*0.1) < abs(c[0]): e1=-e1
        t=Q@e1; lo,hi=t.min(),t.max(); cut=lo+cdef['fraction_from_medial']*(hi-lo)
        lateral=faces[t>cut]; lab[lateral]=nxt
        cutmap[f"cut:{cdef['name']}:medial"]=r; cutmap[f"cut:{cdef['name']}:lateral"]=nxt; nxt+=1
    n=lab.max()+1
    H,U,A=relief(lab)
    reg_group={}; reg_cls={}
    groups=[]
    for gname,g in des['groups'].items():
        mem=[cutmap.get(m,m) if isinstance(m,str) else m for m in g['members']]
        mem=[m for m in mem if A[m]>0]
        for m in mem: reg_group[m]=gname; reg_cls[m]=g['cls']
        faces=np.flatnonzero(np.isin(lab,mem))
        fr=frame_for(faces,g['side']); pv=pivot_for(faces,fr,g['pivot'],g['side'])
        groups.append(dict(name=gname,side=g['side'],cls=g['cls'],parent=g.get('parent','HEAD'),pivot=pv.round(6).tolist(),
                           frame=fr.round(6).tolist(),members=[int(m) for m in mem],designed=True))
    # ---- clothing / body
    cloth=json.load(open(clothing)) if clothing else {"22":"jacket"}
    for k,v in cloth.items(): reg_group[int(k)]='body'; reg_cls[int(k)]=v
    # ---- leftovers -> coarse groups
    Cr=np.stack([np.bincount(lab,weights=ar*Cw[:,k],minlength=n) for k in range(3)],1)/np.maximum(A,1e-12)[:,None]
    gcent={g['name']:np.array(g['pivot']) for g in groups}
    coarse={}
    for r in range(n):
        if A[r]==0 or r in reg_group: continue
        x,y,z=Cr[r]; s='C' if abs(x)<0.1 else ('L' if x>0 else 'R')
        if z< -0.13: gname=f"neck_{'front' if (y<-0.12 and s=='C') else ('back' if y>0.05 else s)}"; parent='NECK'
        elif z>0.62 and y>-0.45: gname=f"crown_{s}"; parent='HEAD'
        elif y>0.08: gname=f"back_head_{s}"; parent='HEAD'
        elif abs(x)>0.33: gname=f"ear_misc_{s}"; parent='HEAD'
        else: gname=f"face_misc_{s}"; parent='HEAD'
        reg_group[r]=gname
        reg_cls[r]='under' if (U[r]<-0.55) else ('neck_plate' if parent=='NECK' else 'plate')
        coarse.setdefault(gname,dict(parent=parent,side=s if s!='C' else 'C',members=[]))['members'].append(r)
    for gname,g in coarse.items():
        faces=np.flatnonzero(np.isin(lab,g['members'])); fr=frame_for(faces,g['side'])
        pv=(Cw[faces]*ar[faces,None]).sum(0)/ar[faces].sum()
        groups.append(dict(name=gname,side=g['side'],cls='mixed',parent=g['parent'],pivot=pv.round(6).tolist(),frame=fr.round(6).tolist(),
                           members=[int(m) for m in g['members']],designed=False))
    body=[r for r,gname in reg_group.items() if gname=='body']
    # ---- controls (expand mirrors)
    gside={g['name']:g['side'] for g in groups}
    def mirror_group(gn):
        if gn.endswith('_L'): return gn[:-2]+'_R'
        if '_L_' in gn: return gn.replace('_L_','_R_')
        return gn
    controls=[]
    for cname,c in des['controls'].items():
        controls.append(dict(name=cname,range=c['range'],neutral=c.get('neutral',0.0),effects=c['effects']))
        if cname in des['mirror_controls']:
            eff=[]
            for e in c['effects']:
                e2=copy.deepcopy(e); g2=mirror_group(e['group'])
                if gside.get(e['group'])=='C': g2=e['group']
                e2['group']=g2
                if gside.get(e['group'])!='C':
                    if 'loc' in e2: e2['loc']=[-e2['loc'][0],e2['loc'][1],e2['loc'][2]]
                    if 'rot' in e2: e2['rot']=[e2['rot'][0],-e2['rot'][1],-e2['rot'][2]]
                eff.append(e2)
            controls.append(dict(name=des['mirror_controls'][cname],range=c['range'],neutral=c.get('neutral',0.0),effects=eff))
    # validate group refs
    gnames=set(gside)
    for c in controls:
        for e in c['effects']: assert e['group'] in gnames, (c['name'],e['group'])
    for hc,rng in (('head_yaw',[-0.7,0.7]),('head_pitch',[-0.45,0.45]),('head_roll',[-0.35,0.35])):
        controls.append(dict(name=hc,range=rng,neutral=0.0,effects=[],head=True))
    # ---- regions (per-plate pivots)
    m=lab[fa]!=lab[fb]; mid=(Vw[ev[m,0]]+Vw[ev[m,1]])/2; w=elen[m]
    B=np.zeros((n,3)); BW=np.zeros(n)
    for side in (lab[fa][m],lab[fb][m]):
        for k in range(3): B[:,k]+=np.bincount(side,weights=w*mid[:,k],minlength=n)
        BW+=np.bincount(side,weights=w,minlength=n)
    Nr=np.stack([np.bincount(lab,weights=ar*FNw[:,k],minlength=n) for k in range(3)],1); Nr/=np.linalg.norm(Nr,axis=1,keepdims=True)+1e-12
    regs=[]; names=set()
    cnt={}
    for r in range(n):
        if A[r]==0: continue
        gname=reg_group[r]; cnt[gname]=cnt.get(gname,0)+1
        name=f"{gname}.{cnt[gname]:02d}"
        regs.append(dict(id=r,name=name,group=gname,cls=reg_cls[r],pivot=(B[r]/max(BW[r],1e-12)).round(6).tolist() if BW[r]>0 else Cr[r].round(6).tolist(),
                         normal=Nr[r].round(5).tolist(),area=round(float(A[r]),6),relief=round(float(U[r]),3)))
    groups.append(dict(name='body',side='C',cls='body',parent='ROOT',pivot=[0,0,-0.6],frame=np.eye(3).tolist(),members=body,designed=False))
    rig=dict(version=1,units='Meshy model units (bust 1.9 tall)',plate_thickness=0.008,
             head=des['head'],groups=groups,controls=controls,expressions=des['expressions'],
             stage=dict(lens=85,cam=[1.55,-4.4,0.62],target=[0.0,0.0,0.22]))
    os.makedirs(out,exist_ok=True)
    chk=float(np.round(V[::max(1,len(V)//997)].sum(),3))
    json.dump({'_checksum':chk,'regions':regs},open(os.path.join(out,'regions.json'),'w'),indent=0)
    json.dump(rig,open(os.path.join(out,'rig.json'),'w'),indent=1)
    with lzma.open(os.path.join(out,'face_labels.u16.xz'),'wb',preset=9|lzma.PRESET_EXTREME) as fh: fh.write(lab.astype(np.uint16).tobytes())
    np.save(os.path.join(out,'labels_final.npy'),lab)
    cp=os.path.join(W,'core_final.npz')
    if os.path.exists(cp):
        c=np.load(cp); bio=io.BytesIO(); np.savez(bio,V=c['V'].astype(np.float32),F=c['F'].astype(np.int32))
        with lzma.open(os.path.join(out,'core.npz.xz'),'wb',preset=9) as fh: fh.write(bio.getvalue())
    des_groups=[g for g in groups if g['designed']]
    print('regions',len(regs),'groups',len(groups),'designed',len(des_groups),'controls',len(controls))
    print('coarse',{g:len(v['members']) for g,v in coarse.items()})
if __name__=='__main__':
    a=sys.argv[1:]; cl=None
    if '--clothing' in a: i=a.index('--clothing'); cl=a[i+1]; a=a[:i]+a[i+2:]
    main(a[0],a[1],a[2],cl)
