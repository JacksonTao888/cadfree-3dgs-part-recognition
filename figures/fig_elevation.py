"""Figure_elevation: screw-b twin rendered at 0/+8/+15/+30/+45 deg elevation (same azimuth), print width 5.17 in."""
import os, numpy as np
from PIL import Image
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"pdf.fonttype":42,"font.size":9})
R=os.path.expanduser("~/recon_runs/C125/renders_elevsweep"); OUT=os.path.expanduser("~/2026_/revision/figs/out")
bands=["elev_base","elev_+8","elev_+15","elev_+30","elev_+45"]; labels=["0$^{\\circ}$ (capture)","+8$^{\\circ}$","+15$^{\\circ}$","+30$^{\\circ}$","+45$^{\\circ}$"]
fig,axs=plt.subplots(1,5,figsize=(5.17,1.35))
for ax,b,l in zip(axs,bands,labels):
    im=np.asarray(Image.open(f"{R}/{b}/0003.png").convert("RGB")); m=im.sum(2)>15; ys,xs=np.where(m)
    y0,y1,x0,x1=max(ys.min()-20,0),ys.max()+20,max(xs.min()-20,0),xs.max()+20
    ax.imshow(im[y0:y1,x0:x1]); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(l,fontsize=9,pad=3)
fig.subplots_adjust(left=0.005,right=0.995,top=0.82,bottom=0.02,wspace=0.05)
for ext in ("pdf","png"): fig.savefig(f"{OUT}/Figure_elevation.{ext}",dpi=300,bbox_inches="tight")
print("saved Figure_elevation")
