"""t-SNE of real-test features: frozen DINOv2 (raw) vs meta-head embedding.
Per-class colors grouped by category (gear/nut/screw) to show within-category separation.
Saves figures/Figure_tsne.{pdf,png}."""
import torch, numpy as np, csv
import sys; sys.path.insert(0,os.environ.get("CADFREE_ROOT", os.path.expanduser("~/cadfree_work"))); from run_meta import feats_for_dir, Head, meta_train, DEV, ROOT
from sklearn.manifold import TSNE
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
                     "svg.fonttype":"none","pdf.fonttype":42,"font.size":9})
torch.manual_seed(0); np.random.seed(0)
allcls=sorted(r["class"] for r in csv.DictReader(open(f"{ROOT}/clip_to_class.csv")))
order=sorted(allcls,key=lambda c:({"gear":0,"nut":1,"screw":2}[c.split("_")[0]],c))
print("caching ...")
model=torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).to(DEV).eval()
real={c:feats_for_dir(model,f"{ROOT}/data_real_test/{c}") for c in allcls}
import os; SRC=os.environ.get("SRC",f"{ROOT}/data_3dgs_source"); CAP=int(os.environ.get("CAP","400")) or None
FIGOUT = os.environ.get("CADFREE_FIGOUT", "figures/out"); os.makedirs(FIGOUT, exist_ok=True)
src={c:feats_for_dir(model,f"{SRC}/{c}",cap=CAP) for c in allcls}
head=Head().to(DEV); meta_train(head,src,allcls,2000,8,5,10,1e-3,0,"nn"); head.eval()

# stack real-test features (cap per class for balance/speed)
raw=[]; emb=[]; lab=[]
with torch.no_grad():
    for c in order:
        f=real[c]
        if f.shape[0]>90: f=f[torch.randperm(f.shape[0])[:90]]
        raw.append(f.numpy()); emb.append(head(f.to(DEV)).cpu().numpy()); lab+=[c]*f.shape[0]
raw=np.concatenate(raw); emb=np.concatenate(emb); lab=np.array(lab)
print("points:",len(lab))
def run_tsne(X): return TSNE(n_components=2,perplexity=30,init="pca",random_state=0,max_iter=1000).fit_transform(X)
print("t-SNE raw ..."); Yr=run_tsne(raw); print("t-SNE meta ..."); Ye=run_tsne(emb)

# per-class colors grouped by category
cmaps={"gear":plt.cm.Greens,"nut":plt.cm.Reds,"screw":plt.cm.Blues}
cats={}
for c in order: cats.setdefault(c.split("_")[0],[]).append(c)
color={}
for cat,members in cats.items():
    for i,c in enumerate(members): color[c]=cmaps[cat](0.45+0.5*i/max(1,len(members)-1))

fig,axs=plt.subplots(1,2,figsize=(5.17,3.0))
for ax,(Y,title) in zip(axs,[(Yr,"Frozen DINOv2 features (raw)"),(Ye,"Meta-head embedding")]):
    for c in order:
        m=lab==c; ax.scatter(Y[m,0],Y[m,1],s=5,color=color[c],alpha=0.7,edgecolors="none",label=({"nut_g":"nut_f","nut_h":"nut_g","nut_i":"nut_h"}.get(c,c)).replace("_"," "))
    ax.set_title(title,fontsize=9.5,fontweight="bold",pad=4); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(True); s.set_color("#999"); s.set_linewidth(0.6)
h,l=axs[0].get_legend_handles_labels()
fig.legend(h,l,loc="lower center",ncol=6,fontsize=8,frameon=False,handletextpad=0.2,columnspacing=0.9,markerscale=1.6,bbox_to_anchor=(0.5,0.0))
fig.tight_layout(rect=[0,0.2,1,1])
for ext in ("png","pdf"): fig.savefig(os.path.join(FIGOUT, f"Figure_tsne.{ext}"),dpi=300,bbox_inches="tight")
print("saved Figure_tsne")
