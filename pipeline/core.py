"""Voxelise the closed Meshy mesh (parity scanline along OBJ-Y columns), compute inside distance, erode by D, marching cubes -> core mesh (OBJ coords)."""
import numpy as np, sys, time
from scipy import ndimage
from skimage import measure
t0=time.time()
d=np.load('mesh.npz'); V=d['V']; F=d['F']
h=float(sys.argv[1]) if len(sys.argv)>1 else 0.006
D=float(sys.argv[2]) if len(sys.argv)>2 else 0.04
lo=V.min(0)-3*h; hi=V.max(0)+3*h
nx,ny,nz=np.ceil((hi-lo)/h).astype(int)+1
print('grid',nx,ny,nz)
# columns along Y (axis1): column centers x_i=lo0+i*h, z_k=lo2+k*h
T=V[F]  # (nF,3,3)
x=T[:,:,0]; z=T[:,:,2]
i0=np.ceil((x.min(1)-lo[0])/h).astype(int); i1=np.floor((x.max(1)-lo[0])/h).astype(int)
k0=np.ceil((z.min(1)-lo[2])/h).astype(int); k1=np.floor((z.max(1)-lo[2])/h).astype(int)
ni=np.maximum(0,i1-i0+1); nk=np.maximum(0,k1-k0+1); cnt=ni*nk
tri=np.repeat(np.arange(len(F)),cnt)
off=np.arange(cnt.sum())-np.repeat(np.cumsum(cnt)-cnt,cnt)
ii=i0[tri]+off%np.maximum(ni[tri],1); kk=k0[tri]+off//np.maximum(ni[tri],1)
px=lo[0]+ii*h; pz=lo[2]+kk*h
A=T[tri]
# barycentric in xz plane
a=A[:,0]; b=A[:,1]; c=A[:,2]
v0x=b[:,0]-a[:,0]; v0z=b[:,2]-a[:,2]; v1x=c[:,0]-a[:,0]; v1z=c[:,2]-a[:,2]; v2x=px-a[:,0]; v2z=pz-a[:,2]
den=v0x*v1z-v1x*v0z
ok=np.abs(den)>1e-18
u=np.where(ok,(v2x*v1z-v1x*v2z)/np.where(ok,den,1),-1); w=np.where(ok,(v0x*v2z-v2x*v0z)/np.where(ok,den,1),-1)
inside=ok&(u>=0)&(w>=0)&(u+w<=1)
yy=a[:,1]+u*(b[:,1]-a[:,1])+w*(c[:,1]-a[:,1])
ii,kk,yy=ii[inside],kk[inside],yy[inside]
print('hits',len(yy),time.time()-t0)
col=ii*nz+kk
o=np.lexsort((yy,col)); col=col[o]; yy=yy[o]
occ=np.zeros((nx,ny,nz),bool)
starts=np.r_[0,np.flatnonzero(np.diff(col))+1]; ends=np.r_[starts[1:],len(col)]
bad=0
for s,e in zip(starts,ends):
    n=e-s
    if n%2: bad+=1; n-=1
    ys=yy[s:s+n]; c0=col[s]; i,k=divmod(c0,nz)
    j=np.ceil((ys-lo[1])/h).astype(int)
    for p in range(0,n,2): occ[i,j[p]:j[p+1]+1,k]=True
print('odd columns',bad,'occ',occ.sum(),time.time()-t0)
dist=ndimage.distance_transform_edt(occ)*h
core=dist>D
core=ndimage.binary_opening(core,iterations=1)
lab,n=ndimage.label(core); sizes=ndimage.sum(core,lab,range(1,n+1)); keep=1+np.argmax(sizes); core=lab==keep
print('core vox',core.sum(),'components',n)
sm=ndimage.gaussian_filter(core.astype(np.float32),1.0)
verts,faces,_,_=measure.marching_cubes(sm,0.5,spacing=(h,h,h))
verts+=lo
print('core mesh',verts.shape,faces.shape,time.time()-t0)
np.savez('core.npz',V=verts,F=faces,D=D,h=h)
np.save('occ_dist.npy',dist.astype(np.float32)); np.save('occ_meta.npy',np.r_[lo,h])
