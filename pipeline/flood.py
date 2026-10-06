import numpy as np, seglib, time, heapq, sys
def build_adj(nF,fa,fb):
    deg=np.bincount(fa,minlength=nF)+np.bincount(fb,minlength=nF)
    ptr=np.r_[0,np.cumsum(deg)]
    nb=np.empty(ptr[-1],np.int64); ed=np.empty(ptr[-1],np.int64)
    src=np.r_[fa,fb]; dst=np.r_[fb,fa]; eid=np.r_[np.arange(len(fa)),np.arange(len(fa))]
    o=np.argsort(src,kind='stable'); nb[:]=dst[o]; ed[:]=eid[o]
    return ptr,nb,ed
def watershed(markers,prio_edge,ptr,nb,ed):
    lab=markers.copy(); nF=len(lab)
    h=[]
    seeds=np.flatnonzero(lab>=0)
    # push boundary of markers
    cnt=0
    ptr_l=ptr.tolist(); nb_l=nb.tolist(); ed_l=ed.tolist(); pe=prio_edge.tolist(); labl=lab.tolist()
    for f in seeds.tolist():
        for k in range(ptr_l[f],ptr_l[f+1]):
            g=nb_l[k]
            if labl[g]<0:
                heapq.heappush(h,(pe[ed_l[k]],cnt,g,labl[f])); cnt+=1
    while h:
        p,_,g,l=heapq.heappop(h)
        if labl[g]>=0: continue
        labl[g]=l
        for k in range(ptr_l[g],ptr_l[g+1]):
            q=nb_l[k]
            if labl[q]<0:
                heapq.heappush(h,(max(p*0.0,pe[ed_l[k]]),cnt,q,l)); cnt+=1
    return np.array(labl)
if __name__=='__main__':
    t0=time.time()
    V,F=seglib.load(); nF=len(F)
    E=np.load('edges.npz'); fa,fb,signed=E['fa'],E['fb'],E['signed']
    cdeg=np.maximum(0,-np.degrees(signed))
    ptr,nb,ed=build_adj(nF,fa,fb)
    markers=np.load(sys.argv[1])
    lab=watershed(markers,cdeg,ptr,nb,ed)
    print('flood',time.time()-t0,(lab<0).sum())
    np.save(sys.argv[2],lab)
    ids,cnt=np.unique(lab,return_counts=True); print('regions',len(ids))
    seglib.render(seglib.label_colors(lab),sys.argv[2].replace('.npy',''),views='front,q34,back,left',res=900)
    seglib.render(seglib.label_colors(lab),sys.argv[2].replace('.npy','')+'_zoom',views='front,q34',res=900,zoom=3.2,center_z=0.33)
