"""Height of every face above the eroded/smoothed inner core -> per-region relative relief vs neighbours."""
import numpy as np, sys, os
from scipy.spatial import cKDTree
W=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def face_height():
    p=os.path.join(W,'face_height.npy')
    if os.path.exists(p): return np.load(p)
    E=np.load(os.path.join(W,'edges.npz')); cen=E['cen']
    c=np.load(os.path.join(W,'core.npz')); t=cKDTree(c['V'])
    h,_=t.query(cen,workers=-1); np.save(p,h); return h
def relief(lab):
    E=np.load(os.path.join(W,'edges.npz')); fa,fb,elen,ar=E['fa'],E['fb'],E['elen'],E['ar']
    h=face_height(); n=lab.max()+1
    A=np.bincount(lab,weights=ar,minlength=n); H=np.bincount(lab,weights=ar*h,minlength=n)/np.maximum(A,1e-12)
    # local comparison: within a 2-ring band around each boundary, mean height on each side
    m=lab[fa]!=lab[fb]; a,b,w=fa[m],fb[m],elen[m]
    la,lb=lab[a],lab[b]
    # region "upper score": length-weighted sign of (H_region - H_neighbour)
    s=np.sign(H[la]-H[lb])
    up=np.bincount(la,weights=w*s,minlength=n)+np.bincount(lb,weights=-w*s,minlength=n)
    P=np.bincount(la,weights=w,minlength=n)+np.bincount(lb,weights=w,minlength=n)
    return H, up/np.maximum(P,1e-12), A
if __name__=='__main__':
    lab=np.load(sys.argv[1]); H,U,A=relief(lab)
    np.savez(sys.argv[2],H=H,U=U,A=A)
    for i in [int(x) for x in sys.argv[3:]]: print(i,'H',round(H[i],4),'upper',round(U[i],2),'area',round(A[i],4))
