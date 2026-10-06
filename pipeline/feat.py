import numpy as np, scipy.sparse as sp, time
d=np.load('mesh.npz'); V=d['V']; F=d['F']
nV=len(V)
t=time.time()
# face normals/areas
e1=V[F[:,1]]-V[F[:,0]]; e2=V[F[:,2]]-V[F[:,0]]
fn=np.cross(e1,e2); fa=np.linalg.norm(fn,axis=1)/2; fnu=fn/(2*fa[:,None]+1e-20)
# vertex normals area weighted
vn=np.zeros_like(V)
for k in range(3): np.add.at(vn,F[:,k],fn)
vn/=np.linalg.norm(vn,axis=1,keepdims=True)+1e-20
va=np.zeros(nV); 
for k in range(3): np.add.at(va,F[:,k],fa/3)
# adjacency
i=np.concatenate([F[:,0],F[:,1],F[:,2],F[:,1],F[:,2],F[:,0]])
j=np.concatenate([F[:,1],F[:,2],F[:,0],F[:,0],F[:,1],F[:,2]])
A=sp.coo_matrix((np.ones(len(i)),(i,j)),shape=(nV,nV)).tocsr(); A.data[:]=1
deg=np.asarray(A.sum(1)).ravel()
print('adj',time.time()-t)
# area weighted smoothing operator: x_new = (A_w x)/sum ; weights = neighbor vertex area
W=A.multiply(va[None,:]).tocsr()
ws=np.asarray(W.sum(1)).ravel()
P=sp.diags(1/ws)@W
P=P.tocsr()
np.savez('feat_base.npz',vn=vn,va=va,fnu=fnu,fa=fa)
sp.save_npz('P.npz',P); sp.save_npz('A.npz',A)
# multi-scale smoothing displacement
X=V.copy()
out={}
checkpoints=[4,16,64,256]
for it in range(1,max(checkpoints)+1):
    X=0.5*X+0.5*(P@X)
    if it in checkpoints:
        # smoothed normal from smoothed surface? use original vn
        disp=((V-X)*vn).sum(1)
        out['d%d'%it]=disp
        print(it,'disp pct',np.percentile(disp,[1,5,25,50,75,95,99]),time.time()-t)
np.savez('disp.npz',**out)
