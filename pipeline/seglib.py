import numpy as np, subprocess, os, colorsys
W=os.path.dirname(os.path.abspath(__file__))
BPY=os.path.join(W,'..','bpyenv','.venv','bin','python')
def load():
    d=np.load(os.path.join(W,'mesh.npz')); return d['V'],d['F']
def render(colors,out,views='front,q34,left,back',res=700,zoom=1.0,center_z=None):
    out=os.path.abspath(out); p=out+'_cols.npy'; np.save(p,np.asarray(colors,np.float32))
    cmd=[BPY,os.path.join(W,'render_colors.py'),'--',os.path.join(W,'mesh.npz'),p,out,views,str(res),str(zoom)]
    if center_z is not None: cmd.append(str(center_z))
    r=subprocess.run(cmd,capture_output=True,text=True)
    if r.returncode: print(r.stdout[-2000:],r.stderr[-2000:])
    return [f'{out}_{v}.png' for v in views.split(',')]
def label_colors(labels,seed=1,special=None):
    labels=np.asarray(labels); n=labels.max()+1
    rng=np.random.default_rng(seed)
    h=rng.random(n); s=0.45+0.5*rng.random(n); v=0.6+0.4*rng.random(n)
    pal=np.array([colorsys.hsv_to_rgb(*x) for x in zip(h,s,v)])
    c=pal[np.clip(labels,0,None)]
    c[labels<0]=(0.08,0.08,0.08)
    if special:
        for k,col in special.items(): c[labels==k]=col
    return c
def scalar_colors(x,lo,hi):
    t=np.clip((x-lo)/(hi-lo),0,1)
    # blue(neg)-white-red(pos)
    c=np.zeros((len(x),3))
    c[:,0]=np.where(t>0.5,1,2*t); c[:,2]=np.where(t<0.5,1,2*(1-t)); c[:,1]=1-2*np.abs(t-0.5)
    return c
PAL=np.array([(0.95,0.3,0.3),(0.3,0.75,0.3),(0.3,0.45,0.95),(0.95,0.85,0.2),(0.85,0.35,0.85),(0.2,0.85,0.85),(0.98,0.6,0.2),(0.6,0.4,0.2),(0.55,0.95,0.55),(0.6,0.6,0.98),(0.98,0.6,0.75),(0.5,0.5,0.5)])
def distinct_colors(lab,fa=None,fb=None):
    if fa is None:
        E=np.load(os.path.join(W,'edges.npz')); fa,fb=E['fa'],E['fb']
    n=lab.max()+1
    m=lab[fa]!=lab[fb]
    a=lab[fa][m]; b=lab[fb][m]
    pairs=np.unique(np.stack([np.minimum(a,b),np.maximum(a,b)],1),axis=0)
    nbrs=[[] for _ in range(n)]
    for x,y in pairs: nbrs[x].append(y); nbrs[y].append(x)
    deg=np.array([len(z) for z in nbrs])
    col=np.full(n,-1)
    rng=np.random.default_rng(0)
    for r in np.argsort(-deg):
        used={col[z] for z in nbrs[r]}
        opts=[k for k in range(len(PAL)) if k not in used]
        col[r]=opts[rng.integers(len(opts))] if opts else rng.integers(len(PAL))
    c=PAL[col[lab]]
    return c
