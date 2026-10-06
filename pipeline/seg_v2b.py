import numpy as np, seglib, sys, time
from rag import merge_pass
E=np.load('edges.npz'); fa,fb,elen,ar=E['fa'],E['fb'],E['elen'],E['ar']
Bz=np.load('barrier.npz'); eb=Bz['eb_max']
lab=np.load(sys.argv[1])
T=float(sys.argv[3]); frac=float(sys.argv[4]); mina=float(sys.argv[5])
lab=merge_pass(lab,fa,fb,elen,eb,ar,T=T,frac=frac,min_area=mina)
np.save(sys.argv[2],lab)
out=sys.argv[2].replace('.npy','')
seglib.render(seglib.label_colors(lab),out,views=sys.argv[6] if len(sys.argv)>6 else 'front,q34,back,left',res=900)
