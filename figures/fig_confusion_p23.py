"""Figure_confusion_P23 as a 2x2 grid at text width (5.17 in), from the saved confusion matrices."""
import os
import numpy as np, csv
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
FIGOUT = os.environ.get("CADFREE_FIGOUT", "figures/out"); os.makedirs(FIGOUT, exist_ok=True)
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"pdf.fonttype":42,"font.size":9})
R=os.environ.get("CADFREE_ROOT", os.path.expanduser("~/cadfree_work")); RELABEL={"nut_g":"nut_f","nut_h":"nut_g","nut_i":"nut_h"}; disp=lambda c:RELABEL.get(c,c).replace("_"," ")
allcls=sorted(r["class"] for r in csv.DictReader(open(f"{R}/clip_to_class.csv")))
order=sorted(allcls,key=lambda c:({"gear":0,"nut":1,"screw":2}[c.split("_")[0]],c))
P={"P2":[c for c in order if c.startswith("screw")],"P3":[c for c in order if c.startswith("nut")]}
fig,axs=plt.subplots(2,2,figsize=(5.17,5.4))
for ax,(n,reg) in zip(axs.ravel(),[("P2","10shot"),("P2","zeroshot"),("P3","10shot"),("P3","zeroshot")]):
    CM=np.load(f"{R}/cm_{n}_{reg}.npy"); acc=np.trace(CM)/CM.sum(); Mn=CM/np.clip(CM.sum(1,keepdims=True),1,None); cls=P[n]; K=len(cls)
    ax.imshow(Mn,cmap="Blues",vmin=0,vmax=1,aspect="equal")
    labs=[disp(c) for c in cls]; ax.set_xticks(range(K)); ax.set_xticklabels(labs,rotation=90,fontsize=8); ax.set_yticks(range(K)); ax.set_yticklabels(labs,fontsize=8)
    ax.set_title(f"{n} ({'screws' if n=='P2' else 'nuts'}) · {'10-shot' if reg=='10shot' else 'zero-shot'}  ({acc*100:.1f}%)",fontsize=9,fontweight="bold",pad=4); ax.tick_params(length=1.5)
    for i in range(K):
        for j in range(K):
            v=Mn[i,j]
            if v>=0.10: ax.text(j,i,f"{int(round(v*100))}",ha="center",va="center",fontsize=7.5,color="white" if v>0.55 else "#222")
fig.subplots_adjust(left=0.09,right=0.99,top=0.96,bottom=0.09,wspace=0.35,hspace=0.45)
for ext in ("png","pdf"): fig.savefig(os.path.join(FIGOUT, f"Figure_confusion_P23.{ext}"),dpi=400,bbox_inches="tight")
print("saved Figure_confusion_P23 2x2")
