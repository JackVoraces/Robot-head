import numpy as np, seglib, time, heapq, sys
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra, connected_components
t0=time.time()
V,F=seglib.load(); nF=len(F)
E=np.load('edges.npz'); fa,fb,signed,elen,ar,cen=E['fa'],E['fb'],E['signed'],E['elen'],E['ar'],E['cen']
cdeg=np.maximum(0,-np.degrees(signed))  # concavity per edge
T_CREASE=float(sys.argv[1]) if len(sys.argv)>1 else 20.0
# face barrier = max concavity of its edges
B=np.zeros(nF); np.maximum.at(B,fa,cdeg); np.maximum.at(B,fb,cdeg)
crease_e=cdeg>T_CREASE
crease_f=np.zeros(nF,bool); crease_f[fa[crease_e]]=True; crease_f[fb[crease_e]]=True
print('crease faces',crease_f.mean())
w=np.linalg.norm(cen[fa]-cen[fb],axis=1)+1e-9
G=sp.coo_matrix((np.r_[w,w],(np.r_[fa,fb],np.r_[fb,fa])),shape=(nF,nF)).tocsr()
# multi-source dijkstra via super node
src=np.flatnonzero(crease_f)
Gs=sp.vstack([sp.hstack([G,sp.csr_matrix((nF,1))]),sp.hstack([sp.csr_matrix(([1e-9]*len(src),([0]*len(src),src)),shape=(1,nF)),sp.csr_matrix((1,1))])]).tocsr()
dist=dijkstra(Gs,directed=True,indices=nF)[:nF]
print('dist done',time.time()-t0, np.percentile(dist,[50,90,99]))
np.save('dist.npy',dist)
levels=[0.0015,0.003,0.006,0.012,0.024]
MINA=[2e-6,5e-6,2e-5,8e-5,3e-4]
def comps(r):
    m=dist>r
    sel=m[fa]&m[fb]
    H=sp.coo_matrix((np.ones(sel.sum()),(fa[sel],fb[sel])),shape=(nF,nF))
    n,lab=connected_components(H,directed=False)
    lab=np.where(m,lab,-1)
    return lab
labs=[comps(r) for r in levels]
print('comps',time.time()-t0)
# hierarchical marker selection
def area_of(mask): return ar[mask].sum()
markers=np.full(nF,-1); nm=0
def recurse(mask,k):
    global nm
    if k+1<len(levels):
        sub=labs[k+1][mask]; ids=np.unique(sub[sub>=0])
        kids=[]
        for i in ids:
            mm=mask&(labs[k+1]==i)
            if area_of(mm)>=MINA[k+1]: kids.append(mm)
        if len(kids)>=2:
            for mm in kids: recurse(mm,k+1)
            return
    markers[mask]=nm; nm+=1
# top level
L0=labs[0]; ids,cnt=np.unique(L0[L0>=0],return_counts=True)
# speed: precompute per-level index lists
order=np.argsort(L0,kind='stable'); 
starts=np.searchsorted(L0[order],ids)
for i,s,c in zip(ids,starts,cnt):
    idx=order[s:s+c]
    if ar[idx].sum()<MINA[0]: continue
    # restrict recursion work to local index set
    mask=np.zeros(nF,bool); mask[idx]=True
    if c<50: markers[idx]=nm; nm+=1; continue
    recurse(mask,0)
print('markers',nm,time.time()-t0)
np.save('markers_v1.npy',markers)
