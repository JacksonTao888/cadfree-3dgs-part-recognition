"""Rotational-symmetry / view-redundancy index per part (R1 Major 3, sub-point 3).
Stage 1 (default): embed the 120 capture-elevation renders of every part with frozen DINOv2, compute the mean cosine
similarity between views whose azimuth differs by >= 30 deg (near-duplicate neighbours excluded). High = the ring of
renders is redundant (rotationally symmetric part). Saves logs/symmetry_index.csv.
Stage 2 (--corr): join with per-class zero-shot accuracy from the regenerated confusion npys, Spearman correlation,
scatter figure -> ~/2026_/revision/figs/out/Figure_symmetry.{pdf,png}.
"""
import os, sys, csv, glob, numpy as np, torch
R=os.path.expanduser("~/recon_runs"); OUT=os.path.expanduser("~/2026_/revision/figs/out")
allcls=sorted((r["class"] for r in csv.DictReader(open(f"{R}/clip_to_class.csv"))),key=lambda c:({"gear":0,"nut":1,"screw":2}[c.split("_")[0]],c))
RELABEL={"nut_g":"nut_f","nut_h":"nut_g","nut_i":"nut_h"}; disp=lambda c:RELABEL.get(c,c).replace("_"," ")
if "--corr" not in sys.argv:
    sys.path.insert(0,R); from run_meta import feats_for_dir, DEV
    model=torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).to(DEV).eval()
    rows=[]
    for c in allcls:
        d=f"{R}/data_3dgs_source/{c}"
        paths=sorted(glob.glob(f"{d}/elev_base_*.jpg")); n=len(paths)
        # feats_for_dir sorts paths itself -> order = azimuth index order
        F=feats_for_dir(model,d+"/",cap=None) if False else None
        import torch.nn.functional as Fn; from PIL import Image; from run_meta import TFM
        fs=[]
        for i in range(0,n,64):
            ims=torch.stack([TFM(Image.open(p).convert("RGB")) for p in paths[i:i+64]]).to(DEV)
            with torch.no_grad(): fs.append(Fn.normalize(model.forward_features(ims)["x_norm_clstoken"],dim=-1).cpu())
        F=torch.cat(fs).numpy(); S=F@F.T
        az=np.arange(n)*360.0/n; dz=np.abs(az[:,None]-az[None,:]); dz=np.minimum(dz,360-dz)
        far=dz>=30; near=(dz>0)&(dz<=6)
        idx=float(S[far].mean()); nn_=float(S[near].mean())
        # also: similarity at 180 deg (two-fold symmetry) and best periodic offset
        s180=float(S[np.abs(dz-180)<=1.5].mean())
        rows.append(dict(cls=c,n=n,redundancy_far30=round(idx,4),adjacent_sim=round(nn_,4),sim_180=round(s180,4)))
        print(rows[-1],flush=True)
    with open(f"{R}/logs/symmetry_index.csv","w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("saved logs/symmetry_index.csv")
else:
    import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
    from scipy.stats import spearmanr, pearsonr
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"pdf.fonttype":42,"font.size":9})
    idx={r["cls"]:float(r["redundancy_far30"]) for r in csv.DictReader(open(f"{R}/logs/symmetry_index.csv"))}
    cm0=np.load(f"{R}/cm_P1_zeroshot.npy"); a0=np.diag(cm0)/cm0.sum(1)*100
    x=np.array([idx[c] for c in allcls]); y=a0
    rs,ps=spearmanr(x,y); rp,pp=pearsonr(x,y)
    print(f"Spearman rho={rs:.3f} p={ps:.3f}; Pearson r={rp:.3f} p={pp:.3f}")
    cat={"gear":"#2E7D32","nut":"#B64342","screw":"#0F4D92"}; MK={"gear":"^","nut":"s","screw":"o"}
    fig,ax=plt.subplots(figsize=(5.17,3.3))
    from adjustText import adjust_text
    texts=[]; fixed=[]
    for c,xi,yi in zip(allcls,x,y):
        ax.scatter(xi,yi,s=46 if c.startswith("gear") else 40,marker=MK[c.split("_")[0]],color=cat[c.split("_")[0]],edgecolors="#222",lw=0.5,zorder=3)
        if c=="screw_c":
            fixed.append(ax.annotate(disp(c),(xi,yi),textcoords="offset points",xytext=(6,-12),ha="left",fontsize=8,color=cat["screw"],zorder=4))
        elif c=="screw_d":
            fixed.append(ax.annotate(disp(c),(xi,yi),textcoords="offset points",xytext=(-6,5),ha="right",fontsize=8,color=cat["screw"],zorder=4))
        elif c=="gear_a":
            fixed.append(ax.annotate(disp(c),(xi,yi),textcoords="offset points",xytext=(-7,4),ha="right",fontsize=8,color=cat["gear"],zorder=4))
        else:
            texts.append(ax.text(xi,yi,disp(c),fontsize=8,color=cat[c.split("_")[0]],zorder=4))
    adjust_text(texts,x=list(x),y=list(y),ax=ax,objects=fixed,expand=(1.3,1.6),arrowprops=dict(arrowstyle="-",color="#999",lw=0.5),min_arrow_len=6)
    ax.set_xlabel("view-redundancy index (mean cosine similarity of renders $\\geq$30$^\\circ$ apart)",fontsize=9)
    ax.set_ylabel("zero-shot per-class accuracy (%)",fontsize=9); ax.grid(ls=":",alpha=0.5); ax.tick_params(labelsize=8.5); ax.set_xlim(0.60,0.87); ax.set_ylim(0,104)
    ax.set_title(f"Spearman $\\rho$ = {rs:.2f} (p = {ps:.2f})",fontsize=9.5)
    from matplotlib.lines import Line2D
    hh=[Line2D([0],[0],marker=MK[k],color="w",markerfacecolor=cat[k],markeredgecolor="#222",markersize=7,lw=0) for k in ("gear","nut","screw")]
    ax.legend(hh,["gears","nuts","screws"],loc="lower left",fontsize=8,frameon=True,framealpha=0.9,edgecolor="#bbb",handletextpad=0.4,borderpad=0.5)
    fig.tight_layout()
    for ext in ("pdf","png"): fig.savefig(f"{OUT}/Figure_symmetry.{ext}",dpi=300,bbox_inches="tight")
    w=[i for i,c in enumerate(allcls) if not c.startswith("gear")]; rsw,psw=spearmanr(x[w],y[w])
    print(f"within screw+nut: Spearman rho={rsw:.3f} p={psw:.3f}")
    with open(f"{R}/logs/symmetry_corr.txt","w") as f: f.write(f"spearman {rs:.3f} {ps:.3f}\npearson {rp:.3f} {pp:.3f}\nspearman_within {rsw:.3f} {psw:.3f}\n")
    print("saved Figure_symmetry")
