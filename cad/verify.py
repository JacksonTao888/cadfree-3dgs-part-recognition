"""Compare each generated STL with the measured proportions of the real part (side-view silhouette)."""
import glob, os, numpy as np, trimesh, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
OUT=os.path.dirname(os.path.abspath(__file__))
# measured from the 3DGS twins (side view): (L/d, dk/d, k/d) for screws; (e/m) for nuts/gears
MEAS={"screw_a":(4.67,1.478,1.027),"screw_d":(4.67,1.699,0.701),"screw_f":(3.94,1.430,0.801),"screw_g":(3.83,1.543,0.741)}
ELONG={"nut_a":1.69,"nut_b":1.28,"nut_ce":1.33,"nut_d":3.49,"nut_f":3.91,"nut_g":1.82,"nut_h":1.31,"gear_a":2.13,"gear_b":1.81}
def silhouette_profile(m, axis=2, nb=300):
    """Orthographic side view: width across the axis, per slice along it."""
    v=m.vertices; t=v[:,axis]; w=np.hypot(v[:,(axis+1)%3],v[:,(axis+2)%3])
    lo,hi=t.min(),t.max(); L=hi-lo; out=np.zeros(nb)
    for k in range(nb):
        s=(t>=lo+k*L/nb)&(t<=lo+(k+1)*L/nb)
        out[k]=2*w[s].max() if s.sum()>2 else np.nan
    return L,out
def steps(L,p):
    p=np.where(np.isnan(p),0,p); core=p[8:292]
    thr=(np.percentile(core,20)+np.percentile(core,90))/2; head=core>thr
    idx=np.where(head)[0]; 
    if not len(idx): return None
    if idx[0]<10:
        k=0
        while k<len(core) and core[k]>thr: k+=1
        run=(0,k)
    else:
        k=len(core)-1
        while k>=0 and core[k]>thr: k-=1
        run=(k+1,len(core))
    d=np.median(core[~head]); dk=np.median(core[run[0]:run[1]]); hl=(run[1]-run[0])/300*L
    return L/d, dk/d, hl/d
print(f"{'part':30s} {'measured (L/d, dk/d, k/d)':30s} {'generated':30s}")
for f in sorted(glob.glob(f"{OUT}/*.stl")):
    m=trimesh.load(f); key=os.path.basename(f).split("__")[0]
    L,p=silhouette_profile(m)
    if key in MEAS:
        g=steps(L,p); mm=MEAS[key]
        print(f"{os.path.basename(f):30s} L/d {mm[0]:.2f} dk/d {mm[1]:.2f} k/d {mm[2]:.2f}   ->  L/d {g[0]:.2f} dk/d {g[1]:.2f} k/d {g[2]:.2f}")
    elif key in ELONG:
        ext=m.extents; el=max(ext)/ext[2] if key.startswith(("nut","gear")) else max(ext)/min(ext)
        print(f"{os.path.basename(f):30s} elongation measured {ELONG[key]:.2f}            ->  {el:.2f}")
