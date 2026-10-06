#!/usr/bin/env python3
"""Apply segmentation edits.  edits.py BASE_LABELS.npy EDITS.json OUT_LABELS.npy
EDITS.json = {
  "split":  [{"id": 12, "seeds": [[x,y,z],[x,y,z],...]}],   # world coords (Blender Z-up). Region re-flooded from seeds along creases.
  "merge":  [{"ids":[a,b,...]} | {"points":[[x,y,z],[x,y,z]]}], # ids refer to BASE labels; points resolved AFTER splits
  "absorb": [{"id": r}]                                     # dissolve region r into best neighbours (per-face flood)
}
Output labels are compacted (renumbered)."""
import sys, os, json, heapq, numpy as np
W=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E=np.load(os.path.join(W,'edges.npz')); fa,fb,cen,ar=E['fa'],E['fb'],E['cen'],E['ar']
eb=np.load(os.path.join(W,'barrier.npz'))['eb_max']; cdeg=np.load(os.path.join(W,'barrier.npz'))['cdeg']
nF=len(ar)
def to_obj(p): p=np.asarray(p,float); return np.array([p[0],p[2],-p[1]])
_adj=None
def adj():
    global _adj
    if _adj is None:
        src=np.r_[fa,fb]; dst=np.r_[fb,fa]; eid=np.r_[np.arange(len(fa)),np.arange(len(fa))]
        o=np.argsort(src,kind='stable'); ptr=np.r_[0,np.cumsum(np.bincount(src,minlength=nF))]
        _adj=(ptr,dst[o],eid[o])
    return _adj
def flood(lab,allowed,prio):
    ptr,nb,ed=adj(); h=[]; cnt=0
    lab=lab.copy()
    seeds=np.flatnonzero(allowed&(lab>=0))
    for f in seeds:
        for k in range(ptr[f],ptr[f+1]):
            g=nb[k]
            if allowed[g] and lab[g]<0: heapq.heappush(h,(prio[ed[k]],cnt,g,lab[f])); cnt+=1
    while h:
        p,_,g,l=heapq.heappop(h)
        if lab[g]>=0: continue
        lab[g]=l
        for k in range(ptr[g],ptr[g+1]):
            q=nb[k]
            if allowed[q] and lab[q]<0: heapq.heappush(h,(prio[ed[k]],cnt,q,l)); cnt+=1
    return lab
def nearest_face(p,mask=None):
    c=cen if mask is None else cen[mask]
    i=np.argmin(((c-to_obj(p))**2).sum(1))
    return i if mask is None else np.flatnonzero(mask)[i]
def apply(lab,ed):
    lab=lab.copy(); nxt=lab.max()+1
    for s in ed.get('split',[]):
        r=s['id']; m=lab==r
        tmp=np.full(nF,-1); newids=[]
        for k,p in enumerate(s['seeds']):
            f=nearest_face(p,m); nid=r if k==0 else nxt+k-1; tmp[f]=nid; newids.append(nid)
        nxt+=len(s['seeds'])-1
        res=flood(tmp,m,eb)
        lab[m]=np.where(res[m]>=0,res[m],r)
    for a in ed.get('absorb',[]):
        m=lab==a['id']; tmp=np.where(m,-1,lab); 
        allowed=np.ones(nF,bool)
        res=flood(tmp,allowed,eb); lab[m]=res[m]
    # merges
    par={}
    def f(x):
        while par.get(x,x)!=x: x=par[x]
        return x
    for mg in ed.get('merge',[]):
        ids=mg.get('ids') or [int(lab[nearest_face(p)]) for p in mg['points']]
        r0=f(ids[0])
        for x in ids[1:]:
            rx=f(x)
            if rx!=r0: par[rx]=r0
    if par:
        u=np.unique(lab); mp={x:f(x) for x in u}; lab=np.vectorize(mp.get)(lab)
    return lab
if __name__=='__main__':
    lab=np.load(sys.argv[1]); ed=json.load(open(sys.argv[2]))
    out=apply(lab,ed)
    u,inv=np.unique(out,return_inverse=True)
    np.save(sys.argv[3],inv.astype(np.int32)); print('regions',len(u))
