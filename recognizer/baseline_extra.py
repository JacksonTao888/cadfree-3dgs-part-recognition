import os, torch, numpy as np, csv, random
from run_matrix import feats_for_dir, ROOT, DEV
import torch.nn.functional as F
SRC=os.environ.get("SRC",f"{ROOT}/data_3dgs_source"); CAP=int(os.environ.get("CAP","400")) or None
allcls=sorted(r["class"] for r in csv.DictReader(open(f"{ROOT}/clip_to_class.csv")))
protos={"P1":allcls,"P2":[c for c in allcls if c.startswith("screw")],"P3":[c for c in allcls if c.startswith("nut")]}

def epi_1nn(src, real, classes, regime, k_shot, k_query, episodes, rng):
    accs=[]
    for _ in range(episodes):
        gals=[]; qf=[]; qy=[]; ok=True
        for i,c in enumerate(classes):
            rf=real[c]
            if rf.shape[0]<k_shot+1: ok=False; break
            idx=torch.randperm(rf.shape[0],generator=rng)
            sup=rf[idx[:k_shot]].to(DEV); qn=min(k_query,rf.shape[0]-k_shot); qry=rf[idx[k_shot:k_shot+qn]].to(DEV)
            gals.append(sup if regime=="support-only" else (src[c].to(DEV) if regime=="source-only" else torch.cat([src[c].to(DEV),sup])))
            qf.append(qry); qy+=[i]*qry.shape[0]
        if not ok: continue
        Q=torch.cat(qf); y=torch.tensor(qy).to(DEV)
        sc=torch.stack([(Q@g.t()).max(1).values for g in gals],1)
        accs.append((sc.argmax(1)==y).float().mean().item())
    return np.array(accs).mean()*100 if accs else float('nan')

def run_backbone(name, embed_src, embed_real):
    print(f"\n=== {name}: no-meta 1-NN (frozen features), 3-seed mean±std ===", flush=True)
    src, real = embed_src(), embed_real()
    for pname,cls in protos.items():
        row=f"  {pname}:"
        for reg,lbl in [("support-only","10shot"),("source-only","zeroshot")]:
            vals=[]
            for s in range(3):
                rng=torch.Generator().manual_seed(s); random.seed(s)
                vals.append(epi_1nn(src,real,cls,reg,10,15,300,rng))
            v=np.array(vals); row+=f"  {lbl}={v.mean():.1f}±{v.std():.1f}"
        print(row, flush=True)

# --- DINOv2 ---
import torch.hub
m=torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).to(DEV).eval()
run_backbone("frozen DINOv2 ViT-B/14",
    lambda: {c:feats_for_dir(m,f"{SRC}/{c}",cap=CAP) for c in allcls},
    lambda: {c:feats_for_dir(m,f"{ROOT}/data_real_test/{c}") for c in allcls})

# --- CLIP (if available) ---
try:
    if os.environ.get('SKIP_CLIP'): raise RuntimeError('skipped by SKIP_CLIP')
    import open_clip
    cm,_,cpre=open_clip.create_model_and_transforms('ViT-B-16',pretrained='openai'); cm=cm.to(DEV).eval()
    from PIL import Image; import glob
    @torch.no_grad()
    def clip_feats(d,batch=64,cap=400):
        ps=sorted(glob.glob(f"{d}/*.jpg")+glob.glob(f"{d}/*.png"))
        if cap and len(ps)>cap: random.Random(0).shuffle(ps); ps=ps[:cap]
        out=[]
        for i in range(0,len(ps),batch):
            ims=torch.stack([cpre(Image.open(p).convert("RGB")) for p in ps[i:i+batch]]).to(DEV)
            f=cm.encode_image(ims); out.append(F.normalize(f,dim=-1).cpu())
        return torch.cat(out) if out else torch.zeros(0,512)
    run_backbone("frozen CLIP ViT-B/16",
        lambda: {c:clip_feats(f"{ROOT}/data_3dgs_source/{c}") for c in allcls},
        lambda: {c:clip_feats(f"{ROOT}/data_real_test/{c}") for c in allcls})
except Exception as e:
    print("CLIP skipped:", repr(e)[:120], flush=True)
print("DONE_BASELINE_EXTRA", flush=True)
