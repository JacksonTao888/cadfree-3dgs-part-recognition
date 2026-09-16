"""Figure_fidelity: real masked frame (video A) vs twin render at the SAME training camera, cropped to the
thread region, for four screws. Needs render_quality.py to have produced <clip>/gs_obj_hi/output/train/ours_98/."""
import os, glob, csv, numpy as np
from PIL import Image
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"pdf.fonttype":42,"font.size":9})
ROOT=os.path.expanduser("~/recon_runs"); OUT=os.path.expanduser("~/2026_/revision/figs/out"); os.makedirs(OUT,exist_ok=True)
c2c={r["class"]:r["clip"] for r in csv.DictReader(open(f"{ROOT}/clip_to_class.csv"))}
screws=["screw_a","screw_b","screw_c","screw_e"]
def pair(clip,frac=0.30):
    d=f"{ROOT}/{clip}/gs_obj_hi/output/train/ours_98"
    gts=sorted(glob.glob(f"{d}/gt/*.png")); rs=sorted(glob.glob(f"{d}/renders/*.png"))
    i=int(frac*len(gts)); G=np.asarray(Image.open(gts[i]).convert("RGB")); R=np.asarray(Image.open(rs[i]).convert("RGB"))
    m=G.sum(2)>6; ys,xs=np.where(m); y0,y1,x0,x1=ys.min(),ys.max(),xs.min(),xs.max()
    # thread region: the half of the object's long axis away from the head is not known a priori; crop the central 60% band
    h=y1-y0; w=x1-x0
    if w>=h: cx0=x0+int(0.35*w); cx1=x0+int(0.95*w); cy0=y0-int(0.05*h); cy1=y1+int(0.05*h)
    else:    cy0=y0+int(0.35*h); cy1=y0+int(0.95*h); cx0=x0-int(0.05*w); cx1=x1+int(0.05*w)
    cy0,cx0=max(cy0,0),max(cx0,0)
    # fixed 1.5:1 aspect so the panels align
    H,Wd=G.shape[:2]; ch,cw=cy1-cy0,cx1-cx0; want=int(cw/1.5)
    if ch<want: pad=(want-ch)//2; cy0=max(0,cy0-pad); cy1=min(H,cy0+want)
    elif ch>want: mid=(cy0+cy1)//2; cy0=mid-want//2; cy1=cy0+want
    return G[cy0:cy1,cx0:cx1],R[cy0:cy1,cx0:cx1]
fig,axs=plt.subplots(2,len(screws),figsize=(5.17,2.0))
for j,c in enumerate(screws):
    G,R=pair(c2c[c])
    for i,(im,lab) in enumerate([(G,"real (video A)"),(R,"3DGS render")]):
        ax=axs[i,j]; ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        if j==0: ax.set_ylabel(lab,fontsize=8.5)
        if i==0: ax.set_title(c.replace("_"," "),fontsize=9,fontweight="bold",pad=3)
fig.subplots_adjust(left=0.07,right=0.995,top=0.9,bottom=0.02,wspace=0.06,hspace=0.08)
for ext in("pdf","png"): fig.savefig(f"{OUT}/Figure_fidelity.{ext}",dpi=300,bbox_inches="tight")
print("saved Figure_fidelity")
