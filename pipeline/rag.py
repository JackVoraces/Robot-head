import numpy as np
class UF:
    def __init__(s,n): s.p=np.arange(n)
    def f(s,x):
        p=s.p
        r=x
        while p[r]!=r: r=p[r]
        while p[x]!=r: p[x],x=r,p[x]
        return r
def boundary_stats(lab,fa,fb,elen,cdeg,T):
    m=lab[fa]!=lab[fb]
    i=lab[fa][m]; j=lab[fb][m]
    lo=np.minimum(i,j); hi=np.maximum(i,j)
    n=lab.max()+1
    key=lo.astype(np.int64)*n+hi
    L=np.bincount(np.unique(key,return_inverse=True)[1],weights=elen[m])
    uk,inv=np.unique(key,return_inverse=True)
    L=np.bincount(inv,weights=elen[m]); Lc=np.bincount(inv,weights=elen[m]*(cdeg[m]>T))
    C=np.bincount(inv,weights=elen[m]*cdeg[m])/L
    return uk//n, uk%n, L, Lc, C
def relabel(lab):
    u,inv=np.unique(lab,return_inverse=True); return inv
def merge_pass(lab,fa,fb,elen,cdeg,ar,T=15,frac=0.35,min_area=3e-4,max_iter=20):
    for it in range(max_iter):
        n=lab.max()+1
        A=np.bincount(lab,weights=ar,minlength=n)
        P=np.bincount(lab[fa][lab[fa]!=lab[fb]],weights=elen[lab[fa]!=lab[fb]],minlength=n)+np.bincount(lab[fb][lab[fa]!=lab[fb]],weights=elen[lab[fa]!=lab[fb]],minlength=n)
        i,j,L,Lc,C=boundary_stats(lab,fa,fb,elen,cdeg,T)
        uf=UF(n); changed=0
        # rule1: mostly non-crease shared boundary that is a meaningful part of the smaller region's perimeter
        sm=np.minimum(P[i],P[j])
        r1=(Lc/L<frac)&(L>0.15*sm)
        # rule2: small region -> best neighbour (longest boundary with lowest concavity)
        score=L*(1-Lc/L+0.05)
        best=np.full(n,-1); bests=np.zeros(n)
        for a,b,s in ((i,j,score),(j,i,score)):
            o=np.argsort(s)  # ascending so last write wins = max
            best[a[o]]=b[o]; bests[a[o]]=s[o]
        order=np.argsort(L)[::-1]
        for k in np.flatnonzero(r1):
            ra,rb=uf.f(i[k]),uf.f(j[k])
            if ra!=rb: uf.p[ra]=rb; changed+=1
        small=np.flatnonzero((A<min_area)&(best>=0))
        for s in small[np.argsort(A[small])]:
            ra,rb=uf.f(s),uf.f(best[s])
            if ra!=rb: uf.p[ra]=rb; changed+=1
        roots=np.array([uf.f(x) for x in range(n)])
        lab=relabel(roots[lab])
        print(' merge iter',it,'changed',changed,'regions',lab.max()+1)
        if changed==0: break
    return lab
