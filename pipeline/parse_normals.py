import numpy as np, sys
path=sys.argv[1]
data=open(path,'rb').read().decode()
lines=data.split('\n')
vn=np.array([l.split()[1:4] for l in lines if l.startswith('vn ')],dtype=np.float32)
fl=[l for l in lines if l.startswith('f ')]
FN=np.array([[int(t.split('/')[2]) for t in l.split()[1:4]] for l in fl],dtype=np.int64)-1
print(vn.shape,FN.shape,FN.min(),FN.max())
np.savez('normals.npz',VN=vn,FN=FN)
