import os
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt, numpy as np, csv
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Liberation Sans","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False,
    "axes.linewidth":0.8,"font.size":8.5})
OUT=os.environ.get("CADFREE_FIGOUT", "figures/out"); R=os.environ.get("CADFREE_ROOT", os.path.expanduser("~/cadfree_work"))
GREY="#C7CDD2"; STEEL="#5B8DB8"; GREEN="#2E7D32"; INK="#222"; RED="#B64342"
prot=["P1\n(18-way)","P2 screw\n(8-way)","P3 nut\n(8-way)"]
# methods: (label, color, 10shot, 0shot, 10sd, 0sd)
M=[("no-meta ProtoNet",GREY,[75.2,72.9,73.2],[61.7,65.7,56.4],None,None),
   ("no-meta 1-NN",STEEL,[87.1,84.7,86.5],[71.8,66.5,74.3],None,None),
   ("meta-head 1-NN (ours)",GREEN,[91.6,91.0,90.7],[75.5,74.0,76.0],[1.4,0.9,1.9],[3.1,1.2,3.8])]
chance=[5.6,12.5,12.5]
fig,axs=plt.subplots(1,2,figsize=(4.75,2.55),sharey=True)
x=np.arange(3); w=0.26
from matplotlib.lines import Line2D
for ax,(reg,idx) in zip(axs,[("10-shot",2),("zero-shot",3)]):
    for k,(lab,col,a10,a0,s10,s0) in enumerate(M):
        vals=[a10,a0][0 if idx==2 else 1]; sds=[s10,s0][0 if idx==2 else 1]
        off=(k-1)*w
        ax.bar(x+off,vals,w,color=col,edgecolor=INK,lw=0.5,label=lab,
               yerr=sds,error_kw=dict(elinewidth=0.8,capsize=2,capthick=0.8,ecolor=INK) if sds else None,zorder=3)
        for xi,v in zip(x+off,vals):
            ax.text(xi,v+1.2,f"{v:.0f}",ha="center",va="bottom",fontsize=7.5,color=INK,zorder=4)
    for i,ch in enumerate(chance):
        ax.plot([x[i]-0.42,x[i]+0.42],[ch,ch],ls=(0,(3,2)),color=RED,lw=1.0,zorder=2)
    ax.set_title(reg,fontsize=9.5,fontweight="bold",pad=3)
    ax.set_xticks(x); ax.set_xticklabels(prot,fontsize=8.5); ax.set_ylim(0,106)
    ax.tick_params(axis="y",labelsize=8.5)
    ax.grid(axis="y",ls=":",alpha=0.45,lw=0.6,zorder=0); ax.tick_params(length=2)
axs[0].set_ylabel("recognition accuracy (%)",fontsize=9)
h,l=axs[0].get_legend_handles_labels()
h.append(Line2D([0],[0],color=RED,ls=(0,(3,2)),lw=1.0)); l.append("chance level")
fig.legend(h,l,loc="upper center",ncol=4,fontsize=8,frameon=False,bbox_to_anchor=(0.5,1.06),columnspacing=1.0,handlelength=1.3,handletextpad=0.4)
fig.tight_layout(rect=[0,0,1,0.93])
for ext in ("png","pdf","svg"): fig.savefig(f"{OUT}/Figure_3_results.{ext}",dpi=600,bbox_inches="tight")
plt.close(fig); print("saved Figure_3_results")

# ---- per-class, print width ----
RELABEL={"nut_g":"nut_f","nut_h":"nut_g","nut_i":"nut_h"}; disp=lambda c:RELABEL.get(c,c).replace("_"," ")
allcls=sorted((r["class"] for r in csv.DictReader(open(f"{R}/clip_to_class.csv"))),key=lambda c:({"gear":0,"nut":1,"screw":2}[c.split("_")[0]],c))
K=len(allcls); cm0=np.load(f"{R}/cm_P1_zeroshot.npy"); cm10=np.load(f"{R}/cm_P1_10shot.npy")
a0=np.diag(cm0)/cm0.sum(1)*100; a10=np.diag(cm10)/cm10.sum(1)*100
catcol={"gear":GREEN,"nut":RED,"screw":"#0F4D92"}; x=np.arange(K); w=0.4
fig,ax=plt.subplots(figsize=(4.75,2.7))
ax.bar(x-w/2,a10,w,label="10-shot",color="#3F76B5",edgecolor=INK,lw=0.4,zorder=3)
ax.bar(x+w/2,a0,w,label="zero-shot",color="#A7D3A0",edgecolor=INK,lw=0.4,zorder=3)
ax.axhline(100/K,ls=(0,(3,2)),color=RED,lw=0.9,zorder=2,label="chance level")
ax.set_xticks(x); ax.set_xticklabels([disp(c) for c in allcls],rotation=90,fontsize=8.5)
for t,c in zip(ax.get_xticklabels(),allcls): t.set_color(catcol[c.split("_")[0]])
ax.set_ylabel("per-class accuracy (%)",fontsize=9); ax.set_ylim(0,104); ax.tick_params(axis="y",labelsize=8.5)
ax.grid(axis="y",ls=":",alpha=0.45,lw=0.6,zorder=0); ax.tick_params(length=2)
ax.legend(fontsize=8,loc="upper center",bbox_to_anchor=(0.5,1.14),ncol=3,frameon=False,handlelength=1.4,columnspacing=1.2)
fig.tight_layout()
for ext in ("png","pdf"): fig.savefig(f"{OUT}/Figure_perclass.{ext}",dpi=600,bbox_inches="tight")
print("saved Figure_perclass")
