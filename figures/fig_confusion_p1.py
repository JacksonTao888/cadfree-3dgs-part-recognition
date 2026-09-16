"""Figure_confusion_P1: two 18x18 panels stacked vertically at text width (5.17 in), from saved matrices."""
import os
import numpy as np, csv
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
FIGOUT = os.environ.get("CADFREE_FIGOUT", "figures/out"); os.makedirs(FIGOUT, exist_ok=True)
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"pdf.fonttype":42,"font.size":9})
R=os.environ.get("CADFREE_ROOT", os.path.expanduser("~/cadfree_work")); RELABEL={"nut_g":"nut_f","nut_h":"nut_g","nut_i":"nut_h"}; disp=lambda c:RELABEL.get(c,c).replace("_"," ")
allcls=sorted(r["class"] for r in csv.DictReader(open(f"{R}/clip_to_class.csv")))
order=sorted(allcls,key=lambda c:({"gear":0,"nut":1,"screw":2}[c.split("_")[0]],c)); K=len(order)
fig,axs=plt.subplots(2,1,figsize=(5.17,7.2))
for ax,reg in zip(axs,["10shot","zeroshot"]):
    CM=np.load(f"{R}/cm_P1_{reg}.npy"); acc=np.trace(CM)/CM.sum(); Mn=CM/np.clip(CM.sum(1,keepdims=True),1,None)
    im=ax.imshow(Mn,cmap="Blues",vmin=0,vmax=1,aspect="equal")
    labs=[disp(c) for c in order]; ax.set_xticks(range(K)); ax.set_xticklabels(labs,rotation=90,fontsize=8); ax.set_yticks(range(K)); ax.set_yticklabels(labs,fontsize=8)
    ax.set_title(f"P1 (18-way) · {'10-shot' if reg=='10shot' else 'zero-shot'}  ({acc*100:.1f}%)",fontsize=9.5,fontweight="bold",pad=5); ax.tick_params(length=1.5)
    cat=[c.split("_")[0] for c in order]; b=[i for i in range(1,K) if cat[i]!=cat[i-1]]
    for x in b: ax.axhline(x-.5,color="#d11",lw=.8); ax.axvline(x-.5,color="#d11",lw=.8)
    for i in range(K):
        for j in range(K):
            v=Mn[i,j]
            if v>=0.10: ax.text(j,i,f"{int(round(v*100))}",ha="center",va="center",fontsize=7,color="white" if v>0.55 else "#222")
    cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.02); cb.set_label("row-normalized fraction",fontsize=8); cb.ax.tick_params(labelsize=8)
fig.subplots_adjust(left=0.12,right=0.98,top=0.97,bottom=0.06,hspace=0.30)
for ext in ("png","pdf"): fig.savefig(os.path.join(FIGOUT, f"Figure_confusion_P1.{ext}"),dpi=400,bbox_inches="tight")
print("saved Figure_confusion_P1 stacked")
