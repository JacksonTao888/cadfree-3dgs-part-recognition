import os, torch, numpy as np, csv, random, glob
import torch.nn.functional as F
from PIL import Image
import open_clip
from run_meta import Head, meta_train, eval_regime, DEV, ROOT
print("loading CLIP ...",flush=True)
cm,_,pre=open_clip.create_model_and_transforms('ViT-B-16',pretrained='openai'); cm=cm.to(DEV).eval()
@torch.no_grad()
def feats(d,b=64,cap=400):
    ps=sorted(glob.glob(f"{d}/*.jpg")+glob.glob(f"{d}/*.png"))
    if cap and len(ps)>cap: random.Random(0).shuffle(ps); ps=ps[:cap]
    out=[]
    for i in range(0,len(ps),b):
        ims=torch.stack([pre(Image.open(p).convert("RGB")) for p in ps[i:i+b]]).to(DEV)
        out.append(F.normalize(cm.encode_image(ims).float(),dim=-1).cpu())
    return torch.cat(out) if out else torch.zeros(0,512)
SRC=os.environ.get("SRC",f"{ROOT}/data_3dgs_source"); CAP=int(os.environ.get("CAP","400")) or None
allcls=sorted(r["class"] for r in csv.DictReader(open(f"{ROOT}/clip_to_class.csv")))
print("embedding ...",flush=True)
src={c:feats(f"{SRC}/{c}",cap=CAP) for c in allcls}
real={c:feats(f"{ROOT}/data_real_test/{c}") for c in allcls}
P={"P1":allcls,"P2":[c for c in allcls if c.startswith("screw")],"P3":[c for c in allcls if c.startswith("nut")]}
res={}
for seed in [0,1,2]:
    head=Head(d=512).to(DEV); meta_train(head,src,allcls,2000,8,5,10,1e-3,seed,"nn")
    rng=torch.Generator().manual_seed(seed+1)
    for n,cls in P.items():
        for reg,l in [("support-only","10shot"),("source-only","zeroshot")]:
            a,_=eval_regime(head,src,real,cls,reg,10,15,300,rng,"nn"); res.setdefault((n,l),[]).append(a)
    print(f"seed {seed} done",flush=True)
print("=== CLIP ViT-B/16 meta-head (1-NN), 3-seed mean±std ===",flush=True)
for n in ["P1","P2","P3"]:
    print(f"  {n}: 10shot={np.mean(res[(n,'10shot')]):.1f}±{np.std(res[(n,'10shot')]):.1f}  zeroshot={np.mean(res[(n,'zeroshot')]):.1f}±{np.std(res[(n,'zeroshot')]):.1f}",flush=True)
print("DONE_CLIP_META",flush=True)
