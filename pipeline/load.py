import numpy as np, sys, time
t=time.time()
path=sys.argv[1]
V=[];F=[];N=[]
with open(path,'rb') as fh:
    data=fh.read().decode()
lines=data.split('\n')
v=[l for l in lines if l.startswith('v ')]
f=[l for l in lines if l.startswith('f ')]
V=np.array([l.split()[1:4] for l in v],dtype=np.float64)
def idx(tok): return int(tok.split('/')[0])
F=np.array([[idx(t) for t in l.split()[1:4]] for l in f],dtype=np.int64)-1
print('loaded',V.shape,F.shape,time.time()-t)
np.savez(sys.argv[2],V=V,F=F)
print(f[:3])
