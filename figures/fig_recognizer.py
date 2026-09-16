import os
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt, numpy as np, glob
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
plt.rcParams['font.family']='sans-serif'; plt.rcParams['font.sans-serif']=['Arial','DejaVu Sans','Liberation Sans']
plt.rcParams['svg.fonttype']='none'; plt.rcParams.update({'pdf.fonttype':42})
R=os.environ.get("CADFREE_ROOT", os.path.expanduser("~/cadfree_work")); OUT=os.environ.get("CADFREE_FIGOUT", "figures/out")
DINO_F="#DCE4EF"; DINO_E="#5B7794"; HEAD_F="#F4B24C"; HEAD_E="#C57518"; HEAD_T="#5A3A00"
GREEN="#2E7D32"; RED="#B64342"; BLUE="#15538C"; INK="#222"; GREY="#666"; SNOW="#5B9BD5"
def sqc(p):
    im=Image.open(p).convert("RGB"); w,h=im.size; s=min(w,h); return im.crop(((w-s)//2,(h-s)//2,(w-s)//2+s,(h-s)//2+s))
fig=plt.figure(figsize=(9.6,5.4)); ax=fig.add_axes([0,0,1,1]); ax.axis("off"); ax.set_xlim(0,100); ax.set_ylim(0,100)
def box(x,y,w,h,txt,fc,ec,tc=INK,fs=12,bold=False,lw=1.6,rs=2.5,z=2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0.3,rounding_size={rs}",fc=fc,ec=ec,lw=lw,zorder=z))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color=tc,fontweight="bold" if bold else "normal",zorder=z+1)
def arr(x0,y0,x1,y1,c=GREY,lw=1.8,ms=14,rad=0,z=3):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",mutation_scale=ms,lw=lw,color=c,connectionstyle=f"arc3,rad={rad}",shrinkA=2,shrinkB=2,zorder=z))
def thumb(fx,fy,fw,fh,path,ec,lw=2.3):
    a=fig.add_axes([fx,fy,fw,fh]); a.imshow(np.asarray(sqc(path))); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_visible(True); s.set_color(ec); s.set_linewidth(lw)
gA=sorted(glob.glob(f"{R}/data_rawframes_source/screw_a/*.jpg")); gB=sorted(glob.glob(f"{R}/data_rawframes_source/screw_b/*.jpg"))
A=[gA[1],gA[45],gA[90],gA[135]]; B=[gB[1],gB[45],gB[90]]; query=sorted(glob.glob(f"{R}/data_real_test/screw_a/*.jpg"))[0]
ax.plot([1.5,98.5],[50,50],color="#D5D5D5",lw=1.0,zorder=0)

# ===================== (a) META-TRAINING =====================
ax.text(1.5,98.5,"(a) Meta-training the head · CAD-free source · no real labels",fontsize=13,fontweight="bold",va="top")
tw,th=0.050,0.088
thumb(0.030,0.775,tw,th,A[0],GREEN); thumb(0.088,0.775,tw,th,A[1],GREEN)
thumb(0.030,0.672,tw,th,B[0],RED);   thumb(0.088,0.672,tw,th,B[1],RED)
ax.text(8.3,64.5,"source episode\n($N$-way $K$-shot)",ha="center",va="top",fontsize=10,color=INK)
arr(15.0,75,20.5,75)
box(20.5,68,15,14.5,"Frozen\nDINOv2\nViT-B/14",DINO_F,DINO_E,tc="#2B3A4A",fs=11.5)
ax.text(28,83.2,"❄",ha="center",fontsize=14,color=SNOW,zorder=4)
ax.text(28,66,"frozen —\nnot trained",ha="center",va="top",fontsize=9.5,color=GREY,style="italic")
arr(35.5,75,41.5,75); ax.text(38.5,77,"768-d",ha="center",fontsize=9.5,color=GREY)
box(41.5,64,24,21,"$\\nabla_\\theta$\nMeta-trained head\n(2-layer MLP)",HEAD_F,HEAD_E,tc=HEAD_T,fs=13,bold=True,lw=2.2,rs=3)
ax.text(53.5,62,"the only trained part",ha="center",va="top",fontsize=10,color=HEAD_E,fontweight="bold")
arr(65.5,75,70.5,75)
ax.add_patch(FancyBboxPatch((70.5,61.5),27,27,boxstyle="round,pad=0.3,rounding_size=2.5",fc="white",ec="#9AA6B2",lw=1.4,zorder=2))
ax.text(84,86.8,"learned embedding",ha="center",fontsize=10.5,color=INK,zorder=3)
rng=np.random.RandomState(3)
for cx,cy,col in [(77.5,80,GREEN),(90.5,69,RED)]:
    pts=rng.randn(9,2)*[1.7,1.5]+[cx,cy]; ax.scatter(pts[:,0],pts[:,1],s=30,color=col,edgecolors="white",lw=0.7,zorder=4)
ax.text(82,83.5,"Class A",color=GREEN,fontsize=10,fontweight="bold",zorder=5)
ax.text(86,64.8,"Class B",color=RED,fontsize=10,fontweight="bold",zorder=5)
ax.scatter([80],[73.5],marker="*",s=140,color=BLUE,edgecolors="white",lw=1.0,zorder=6)
ax.text(76.5,73,"query",color=BLUE,fontsize=10,fontweight="bold",ha="right",zorder=6)
ax.text(84,59.5,"soft-nearest-view loss\n(log-sum-exp over each class's views)",ha="center",va="top",fontsize=9.5,color=INK)
ax.add_patch(FancyArrowPatch((81,89),(54,85.5),arrowstyle="-|>",mutation_scale=13,lw=1.7,color=GREY,connectionstyle="arc3,rad=-0.30",zorder=3))
ax.text(64,92,"$\\nabla_\\theta$ updates the head only",ha="center",fontsize=10,color=INK)

# ===================== (b) TEST =====================
ax.text(1.5,46.5,"(b) Test · everything frozen · nearest reference view (1-NN)",fontsize=13,fontweight="bold",va="top")
thumb(0.030,0.150,0.070,0.122,query,INK,2.2); ax.text(8.2,11.5,"real query",ha="center",va="top",fontsize=10,color=INK)
arr(12.8,24,17.0,24)
box(17.0,17,13,14,"Frozen\nDINOv2 ❄",DINO_F,DINO_E,tc="#2B3A4A",fs=11)
arr(30.5,24,34.5,24)
box(34.5,17,13,14,"$\\nabla_\\theta$\nhead\n(frozen)",HEAD_F,HEAD_E,tc=HEAD_T,fs=11,bold=True)
arr(47.8,24,51.8,24)
ax.scatter([53.2],[24],s=66,color=BLUE,edgecolors="white",lw=1.0,zorder=6); ax.text(53.2,19.7,"query\nembedding",ha="center",va="top",fontsize=9,color=INK)
gtw,gth=0.064,0.112; fxs=[0.585,0.675,0.765]
cA=[]; cB=[]
for k,fx in enumerate(fxs):
    thumb(fx,0.255,gtw,gth,A[k],GREEN); cA.append((fx*100,0.255*100+gth*50))
    thumb(fx,0.035,gtw,gth,B[k],RED);   cB.append((fx*100,0.035*100+gth*50))
ax.text(fxs[1]*100+gtw*50,43.5,"Class A",ha="center",color=GREEN,fontsize=10.5,fontweight="bold")
ax.text(fxs[1]*100+gtw*50,1.6,"Class B",ha="center",va="top",color=RED,fontsize=10.5,fontweight="bold")
qn=(54.6,24)
for i,(cx,cy) in enumerate(cA+cB):
    near=(i==0); ax.add_patch(FancyArrowPatch(qn,(cx-gtw*50,cy),arrowstyle="-",connectionstyle="arc3,rad=0.05",lw=3.0 if near else 0.9,color=GREEN if near else "#CBCBCB",zorder=4 if near else 1))
ax.text(57,40,"nearest view (1-NN)",color=GREEN,fontsize=10.5,fontweight="bold",ha="left")
arr(cA[2][0]+gtw*100+0.5,cA[2][1],84.8,28,c=GREEN,lw=2.6,ms=16)
box(84.8,22.5,13,10,"Predict:\nClass A","#E3F1E3",GREEN,tc=GREEN,fs=12,bold=True,rs=2.5)
for ext in ("svg","pdf","png"): fig.savefig(f"{OUT}/Figure_recognizer.{ext}",dpi=600,bbox_inches="tight")
fig.savefig("/tmp/fig2_ours.png",dpi=120,bbox_inches="tight"); print("saved")
