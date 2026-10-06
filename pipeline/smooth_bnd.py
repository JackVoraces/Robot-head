import numpy as np
def tooth_smooth(lab,fa,fb,nF,iters=4,protect=None):
    """Flip faces that have >=2 edge-neighbours sharing one other label (removes 1-triangle sawteeth)."""
    # neighbour table (each face has exactly 3 neighbours in a closed manifold mesh)
    src=np.r_[fa,fb]; dst=np.r_[fb,fa]
    o=np.argsort(src,kind='stable'); N=dst[o].reshape(nF,3)
    lab=lab.copy(); tot=0
    for it in range(iters):
        L=lab[N]; a,b,c=L[:,0],L[:,1],L[:,2]
        cand=np.where((a==b),a,np.where(a==c,a,np.where(b==c,b,-1)))
        flip=(cand>=0)&(cand!=lab)
        if protect is not None: flip&=~protect
        # avoid simultaneous flips of adjacent faces creating new teeth: only flip faces with even index this pass
        sel=flip&((np.arange(nF)+it)%2==0) if it<iters-1 else flip
        lab[sel]=cand[sel]; tot+=sel.sum()
        if sel.sum()==0 and it>0: break
    return lab,tot
if __name__=='__main__':
    import sys
    E=np.load('edges.npz'); fa,fb,ar=E['fa'],E['fb'],E['ar']
    lab=np.load(sys.argv[1]); nF=len(lab)
    n0=(lab[fa]!=lab[fb]).sum()
    lab2,t=tooth_smooth(lab,fa,fb,nF,iters=6)
    print('flipped',t,'boundary edges',n0,'->',(lab2[fa]!=lab2[fb]).sum(),'regions',len(np.unique(lab2)))
    np.save(sys.argv[2],lab2)
