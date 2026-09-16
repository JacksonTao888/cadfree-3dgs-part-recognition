"""Publication-grade figures for revised_manuscript_machines_v3.md (MDPI Machines).
Python/matplotlib only. Editable-SVG text, restrained palette, PNG+SVG+PDF export.
Numbers match v3 exactly. Outputs to $CADFREE_FIGOUT (default figures/out).
"""
import os
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import image as mpimg
import numpy as np, glob, os
from PIL import Image

# ---- mandatory editable-SVG + style ----
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({"axes.spines.right": False, "axes.spines.top": False,
                     "axes.linewidth": 0.8, "legend.frameon": False, "font.size": 10.8})

PAL = {"blue":"#0F4D92","blue2":"#3775BA","green":"#2E7D32","green_s":"#8BCF8B",
       "red":"#B64342","red_s":"#E9A6A1","grey":"#CFCECE","grey_m":"#767676",
       "grey_d":"#4D4D4D","ink":"#272727","teal":"#42949E"}
CAT = {"gear":"#2E7D32","nut":"#B64342","screw":"#0F4D92"}
R = os.environ.get("CADFREE_ROOT", os.path.expanduser("~/cadfree_work"))
OUT = os.environ.get("CADFREE_FIGOUT", "figures/out"); os.makedirs(OUT, exist_ok=True)

def save(fig, name, dpi=600):
    for ext in ("png","svg","pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig); print("saved", name)

def thumb(ax, path, crop_square=True):
    im = Image.open(path).convert("RGB")
    if crop_square:
        w,h = im.size; s = min(w,h); im = im.crop(((w-s)//2,(h-s)//2,(w-s)//2+s,(h-s)//2+s))
    ax.imshow(np.asarray(im)); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(PAL["grey_m"]); sp.set_linewidth(0.8)

def first(p): return sorted(glob.glob(p))[0]

# ============================================================ FIG 1: pipeline
def fig1_pipeline():
    fig = plt.figure(figsize=(7.2, 3.5)); ax = fig.add_axes([0,0,1,1]); ax.axis("off")
    ax.set_xlim(0,100); ax.set_ylim(0,100)
    def box(x,y,w,h,text,fc="white",ec=PAL["grey_d"],tc=PAL["ink"],fs=10.1,bold=False):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.6,rounding_size=2",
                     fc=fc,ec=ec,lw=1.1,zorder=2))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,color=tc,
                fontweight="bold" if bold else "normal",zorder=3)
    def arrow(x0,y0,x1,y1,c=PAL["grey_d"]):
        ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",mutation_scale=11,
                     lw=1.3,color=c,zorder=1,shrinkA=2,shrinkB=2))
    def tn(path,x,y,w,h,label,lc=PAL["grey_d"]):
        a = fig.add_axes([x/100,y/100,w/100,h/100]); thumb(a,path)
        ax.text(x+w*50/100,y-1.5,label,ha="center",va="top",fontsize=8.5,color=lc,
                transform=fig.transFigure if False else ax.transData)
    # --- band labels ---
    ax.add_patch(FancyBboxPatch((1,55),98,43,boxstyle="round,pad=0,rounding_size=2",
                 fc="#F4F7FB",ec="none",zorder=0))
    ax.add_patch(FancyBboxPatch((1,2),98,30,boxstyle="round,pad=0,rounding_size=2",
                 fc="#FBF4F4",ec="none",zorder=0))
    ax.text(2.5,95,"SOURCE  —  built once from video A  (no CAD, no manual labels)",
            fontsize=10.1,color=PAL["blue"],fontweight="bold",va="top")
    ax.text(2.5,29.5,"TARGET  —  recognize real parts from a separate video B",
            fontsize=10.1,color=PAL["red"],fontweight="bold",va="top")
    # --- source row thumbnails ---
    tn(f"{R}/C124/images_hi/frame_0090.jpg", 4,68,13,13,"turntable video A")
    tn(f"{R}/C124/masks_screw_hi/frame_0090.jpg.png", 22,68,13,13,"SAM2 mask")
    tn(f"{R}/twins/C124_K60/renders_final/elev_base/0060.png", 40,72,12,12,"3DGS twin render")
    tn(first(f"{R}/data_rawframes_source/screw_a/*.jpg"), 40,57,12,12,"raw masked frames")
    arrow(17,74.5,22,74.5); arrow(35,74.5,39.5,76)        # video->mask->3dgs
    ax.add_patch(FancyArrowPatch((35,73),(39.5,64),arrowstyle="-|>",mutation_scale=11,lw=1.3,color=PAL["grey_d"]))
    ax.text(19.5,64.5,"COLMAP poses (fixed intrinsics)",ha="center",va="top",fontsize=8.1,color=PAL["grey_m"])
    ax.text(46.5,87.5,"two interchangeable\nsource forms",ha="center",fontsize=8.1,color=PAL["grey_m"],style="italic")
    # --- recognizer (center-right) ---
    box(60,46,22,16,"Frozen DINOv2 (no fine-tuning)\n+ 1-NN meta-head\n(meta-trained on source)",
        fc="#EAF1FB",ec=PAL["blue"],tc=PAL["ink"],fs=9.5,bold=False)
    arrow(52.5,74,60,58); arrow(52.5,63,60,56)            # both sources -> recognizer (train)
    # --- target row ---
    tn(first(f"{R}/data_real_test/screw_a/*.jpg"), 6,9,13,13,"real photo (video B)",lc=PAL["red"])
    arrow(19.5,15.5,60,48, c=PAL["red"])                  # test -> recognizer
    box(85.5,49,13,10,"screw_a",fc="#EAF6EA",ec=PAL["green"],tc=PAL["green"],fs=12.2,bold=True)
    ax.text(92,47.3,"predicted class",ha="center",va="top",fontsize=7.8,color=PAL["grey_m"],style="italic")
    arrow(82,54,85.5,54)
    ax.text(50,38.5,"leakage control: video A (twin) and video B (test) are disjoint captures",
            ha="center",fontsize=8.4,color=PAL["grey_m"],style="italic")
    save(fig,"Figure_1_pipeline")

# ============================================================ FIG 2: dataset montage
def fig2_dataset():
    import csv
    cls=[r["class"] for r in csv.DictReader(open(f"{R}/clip_to_class.csv"))]
    order=sorted(cls,key=lambda c:({"gear":0,"nut":1,"screw":2}[c.split("_")[0]],c))
    relab=lambda c:({"nut_g":"nut_f","nut_h":"nut_g","nut_i":"nut_h"}.get(c,c)).replace("_"," ")
    cols,prows=6,3
    fig=plt.figure(figsize=(4.65,5.15))
    outer=fig.add_gridspec(prows,1,hspace=0.30,top=0.965,bottom=0.045,left=0.06,right=0.995)
    inner=[outer[r].subgridspec(2,cols,hspace=0.06,wspace=0.10) for r in range(prows)]
    def show(ax,path,col):
        im=Image.open(path).convert("RGB"); w,h=im.size; sdim=min(w,h)
        im=im.crop(((w-sdim)//2,(h-sdim)//2,(w-sdim)//2+sdim,(h-sdim)//2+sdim))
        ax.imshow(np.asarray(im)); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(col); sp.set_linewidth(1.2)
    real_p=lambda c: sorted(glob.glob(f"{R}/data_rawframes_source/{c}/*.jpg"))[len(glob.glob(f"{R}/data_rawframes_source/{c}/*.jpg"))//2]
    def rend_p(c):
        fs=sorted(glob.glob(f"{R}/data_3dgs_source/{c}/elev_base_*.jpg")) or sorted(glob.glob(f"{R}/data_3dgs_source/{c}/*.jpg"))
        return fs[len(fs)//3]
    for i,c in enumerate(order):
        pr,pc=i//cols,i%cols; col=CAT[c.split("_")[0]]
        ar=fig.add_subplot(inner[pr][0,pc]); ag=fig.add_subplot(inner[pr][1,pc])
        show(ar,real_p(c),col); show(ag,rend_p(c),col)
        ar.set_title(relab(c),color=col,fontsize=9,fontweight="bold",pad=2.5)
        if pc==0:
            ar.set_ylabel("real",fontsize=8,color=PAL["grey_d"]); ag.set_ylabel("3DGS",fontsize=8,color=PAL["grey_d"])
    for lbl,col,x in [("2 gears",CAT["gear"],0.2),("8 nuts",CAT["nut"],0.5),("8 screws",CAT["screw"],0.8)]:
        fig.text(x,0.006,lbl,color=col,fontsize=9,fontweight="bold",ha="center")
    save(fig,"Figure_2_dataset")

# ============================================================ FIG 3: main results
def fig3_results():
    prot=["P1\n(18-way)","P2 screw\n(8-way)","P3 nut\n(8-way)"]
    base10=[75.2,72.9,73.2]; base0=[61.7,65.7,56.4]
    ours10=[91.6,91.0,90.7]; ours10sd=[1.4,0.9,1.9]
    ours0=[75.5,74.0,76.0];  ours0sd=[3.1,1.2,3.8]
    chance=[5.6,12.5,12.5]
    series=[("no-meta baseline, 10-shot",base10,None,PAL["grey"]),
            ("no-meta baseline, zero-shot",base0,None,PAL["grey_m"]),
            ("1-NN meta-head, 10-shot",ours10,ours10sd,PAL["blue"]),
            ("1-NN meta-head, zero-shot",ours0,ours0sd,PAL["green"])]
    fig,ax=plt.subplots(figsize=(7.0,3.7)); x=np.arange(3); w=0.2
    for k,(name,acc,sd,col) in enumerate(series):
        off=(k-1.5)*w
        ax.bar(x+off,acc,w,yerr=sd,label=name,color=col,edgecolor=PAL["ink"],lw=0.7,
               error_kw=dict(elinewidth=1.0,capthick=1.0,capsize=2.5,ecolor=PAL["ink"]))
    for i,ch in enumerate(chance):
        ax.plot([x[i]-0.46,x[i]+0.46],[ch,ch],ls=(0,(4,2)),color=PAL["red"],lw=1.0,zorder=5)
    ax.text(x[0]-0.46,7.3,"chance",color=PAL["red"],fontsize=8.5,va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(prot); ax.set_ylabel("recognition accuracy (%)")
    ax.set_ylim(0,100); ax.grid(axis="y",ls=":",alpha=0.4,lw=0.6)
    ax.legend(fontsize=9.2,ncol=2,loc="upper center",bbox_to_anchor=(0.5,1.17),columnspacing=1.2,handlelength=1.3)
    fig.tight_layout(); save(fig,"Figure_3_results")

# ============================================================ FIG 4: crossover
def fig4_crossover():
    K=[30,60,180]; xt=[0,1,2]
    data={"P1 (6-way pilot)":dict(raw0=[84.9,90.7,95.5],g0=[80.2,87.3,81.5],
                                  raw10=[93.7,95.7,97.5],g10=[91.0,92.1,96.2]),
          "P2 (3 pilot screws)":dict(raw0=[71.3,82.5,92.7],g0=[67.4,79.0,65.9],
                                  raw10=[88.5,92.0,96.1],g10=[82.0,85.4,92.4])}
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.4),sharey=True)
    for ax,(name,d) in zip(axs,data.items()):
        ax.plot(xt,d["raw10"],ls="--",marker="s",ms=5,color=PAL["red"],lw=1.6,label="raw, 10-shot")
        ax.plot(xt,d["g10"], ls="--",marker="s",ms=5,color=PAL["green"],lw=1.6,label="3DGS, 10-shot")
        ax.plot(xt,d["raw0"],ls="-", marker="o",ms=5,color=PAL["red"],lw=1.8,label="raw, zero-shot")
        ax.plot(xt,d["g0"],  ls="-", marker="o",ms=5,color=PAL["green"],lw=1.8,label="3DGS, zero-shot")
        ax.set_title(name,fontsize=11.5); ax.set_xticks(xt); ax.set_xticklabels(["30","60","180\n(full)"])
        ax.set_xlabel("capture budget K (frames):  sparse → full"); ax.grid(ls=":",alpha=0.4,lw=0.6)
    axs[0].set_ylabel("zero-shot / 10-shot accuracy (%)")
    axs[1].legend(fontsize=8.9,loc="lower right",handlelength=2.0)
    axs[0].annotate("raw ≥ 3DGS at every K\n(no crossover)",xy=(0.5,67),fontsize=8.9,
                    color=PAL["grey_d"],style="italic",ha="center")
    fig.suptitle("When is reconstruction worth it?  Raw frames vs the 3DGS source",
                 fontsize=11.5,fontweight="bold",y=1.0)
    fig.text(0.5,-0.02,"K=30/60: fresh 7k-iter twins (clean sparse points);  K=180: production twin (full capture).  "
             "Frozen-DINOv2 1-NN meta-head.",ha="center",fontsize=8.1,style="italic",color=PAL["grey_m"])
    fig.tight_layout(rect=[0,0.02,1,0.96]); save(fig,"Figure_4_crossover")

fig2_dataset()
print("ALL FIGURES DONE ->", OUT)
