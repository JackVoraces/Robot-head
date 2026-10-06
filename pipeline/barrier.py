import numpy as np
E=np.load('edges.npz'); fa,fb,signed=E['fa'],E['fb'],E['signed']
nF=E['ar'].shape[0]
cdeg=np.maximum(0,-np.degrees(signed))
B=np.zeros(nF); np.maximum.at(B,fa,cdeg); np.maximum.at(B,fb,cdeg)
B1=B.copy(); np.maximum.at(B1,fa,B[fb]); np.maximum.at(B1,fb,B[fa])
# also an integrated measure: sum of concave angles of the face's edges + neighbours' (approx turn across a fillet)
S=np.zeros(nF); np.add.at(S,fa,cdeg); np.add.at(S,fb,cdeg)
S1=S.copy(); np.add.at(S1,fa,S[fb]); np.add.at(S1,fb,S[fa])
eb_max=np.maximum(B1[fa],B1[fb])
np.savez('barrier.npz',B=B,B1=B1,S1=S1,eb_max=eb_max,cdeg=cdeg)
print(np.percentile(eb_max,[10,50,75,90,95]))
