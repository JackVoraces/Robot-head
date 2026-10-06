import numpy as np, seglib, time
V,F=seglib.load(); nF=len(F)
t=time.time()
# edges: build face adjacency via sorted edge keys
E=np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]])
fid=np.tile(np.arange(nF),3)
Es=np.sort(E,1); key=Es[:,0].astype(np.int64)*len(V)+Es[:,1]
o=np.argsort(key,kind='stable'); ks=key[o]
same=ks[1:]==ks[:-1]
assert same.sum()*2==len(ks), (same.sum(), len(ks))  # watertight manifold
a=o[:-1][same]; b=o[1:][same]
fa,fb=fid[a],fid[b]
ev=Es[a]  # edge vertex pair
e1=V[F[:,1]]-V[F[:,0]]; e2=V[F[:,2]]-V[F[:,0]]
fn=np.cross(e1,e2); ar=np.linalg.norm(fn,axis=1); fn/= (ar[:,None]+1e-20); ar/=2
cen=V[F].mean(1)
cosang=np.clip((fn[fa]*fn[fb]).sum(1),-1,1); ang=np.arccos(cosang)
# concave if centroid of fb lies above plane of fa (along fa normal)
s=((cen[fb]-cen[fa])*fn[fa]).sum(1)
signed=np.where(s>0,-ang,ang)  # negative = concave
elen=np.linalg.norm(V[ev[:,0]]-V[ev[:,1]],axis=1)
np.savez('edges.npz',fa=fa,fb=fb,ev=ev,signed=signed,elen=elen,fn=fn,ar=ar,cen=cen)
print('edges',len(fa),time.time()-t)
print('signed deg pct',np.percentile(np.degrees(signed),[0.5,1,2,5,10,50,90,95,98,99,99.5]))
# per-face max concavity
fc=np.zeros(nF); np.maximum.at(fc,fa,-signed); np.maximum.at(fc,fb,-signed)
col=np.tile([[0.9,0.9,0.9]],(nF,1))
deg=np.degrees(fc)
col[deg>15]=(0.3,0.5,1.0)
col[deg>30]=(0.0,0.1,0.9)
col[deg>60]=(0.6,0,0.6)
seglib.render(col,'viz_concave',views='front,q34,back',res=900)
seglib.render(col,'viz_concave_zoom',views='front,q34',res=900,zoom=3.2,center_z=0.33)
