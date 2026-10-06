import numpy as np, sys, json, os
W=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def to_world(a): a=np.asarray(a); return np.stack([a[...,0],-a[...,2],a[...,1]],-1)
def metrics(lab):
    E=np.load(os.path.join(W,'edges.npz')); fa,fb,elen,ar,fn,cen=E['fa'],E['fb'],E['elen'],E['ar'],E['fn'],E['cen']
    d=np.load(os.path.join(W,'mesh.npz')); F=d['F']
    D=np.load(os.path.join(W,'disp.npz'))
    n=lab.max()+1
    A=np.bincount(lab,weights=ar,minlength=n)
    # face-level relief from vertex displacement (64 & 256 iterations)
    out={}
    for k in ('d64','d256'):
        fd=D[k][F].mean(1); out[k]=np.bincount(lab,weights=ar*fd,minlength=n)/np.maximum(A,1e-12)
    # smoothed normal per face: region-independent, use 256-smoothed displacement direction? use area-weighted normal of 2-ring: approximate with region normal of neighbours
    # step direction: across each boundary edge, height of neighbour face centroid relative to this face's plane
    m=lab[fa]!=lab[fb]
    a,b=fa[m],fb[m]; w=elen[m]
    hb=((cen[b]-cen[a])*fn[a]).sum(1)   # >0: neighbour lies above a's plane (a is lower / in a recess)
    ha=((cen[a]-cen[b])*fn[b]).sum(1)
    up=np.bincount(lab[a],weights=w*np.sign(hb),minlength=n)+np.bincount(lab[b],weights=w*np.sign(ha),minlength=n)
    P=np.bincount(lab[a],weights=w,minlength=n)+np.bincount(lab[b],weights=w,minlength=n)
    out['recess']=up/np.maximum(P,1e-12)  # +1 = every neighbour is above -> deep recess; -1 = raised plate
    out['area']=A; out['perim']=P
    C=np.stack([np.bincount(lab,weights=ar*cen[:,k],minlength=n) for k in range(3)],1)/np.maximum(A,1e-12)[:,None]
    out['centroid']=to_world(C)
    return out
if __name__=='__main__':
    lab=np.load(sys.argv[1]); M=metrics(lab)
    np.savez(sys.argv[2],**M)
    ids=[int(x) for x in sys.argv[3:]]
    for i in ids: print(i,{k:(np.round(v[i],4).tolist() if np.ndim(v[i]) else round(float(v[i]),4)) for k,v in M.items()})
