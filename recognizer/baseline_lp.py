"""Linear-probe baseline: logistic regression on frozen DINOv2 features, 10 real shots.
Answers the 'did you train a classifier / fine-tune on the real support?' reviewer ask.
Same protocol as run_meta.py: 10-shot, 15 query, 300 episodes, 3 seeds; P1/P2/P3."""
import os
import glob, csv, os, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
ROOT=os.environ.get("CADFREE_ROOT", os.path.expanduser("~/cadfree_work")); DEV="cuda" if torch.cuda.is_available() else "cpu"
TFM=transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),transforms.ToTensor(),
                        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
@torch.no_grad()
def feats(model,d,batch=64,cap=400):
    ps=sorted(glob.glob(f"{d}/*.jpg")+glob.glob(f"{d}/*.png"))
    if cap and len(ps)>cap: import random as r; r.Random(0).shuffle(ps); ps=ps[:cap]
    out=[]
    for i in range(0,len(ps),batch):
        ims=torch.stack([TFM(Image.open(p).convert("RGB")) for p in ps[i:i+batch]]).to(DEV)
        f=model.forward_features(ims)["x_norm_clstoken"]
        out.append(F.normalize(f,dim=-1).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0,768),np.float32)
cls=sorted(r["class"] for r in csv.DictReader(open(f"{ROOT}/clip_to_class.csv")))
print("classes:",len(cls))
model=torch.hub.load("facebookresearch/dinov2","dinov2_vitb14",trust_repo=True).to(DEV).eval()
cache=f"{ROOT}/_real_feats_b14.npz"
if os.path.exists(cache):
    z=np.load(cache,allow_pickle=True); real={c:z[c] for c in cls}
else:
    real={c:feats(model,f"{ROOT}/data_real_test/{c}") for c in cls}
    np.savez(cache,**real)
print("real feats per class:",{c:real[c].shape[0] for c in cls[:3]},"...")
protos={"P1_all(18)":cls,"P2_screw(8)":[c for c in cls if c.startswith("screw")],"P3_nut(8)":[c for c in cls if c.startswith("nut")]}
K,Q,EP=10,15,300
res={p:[] for p in protos}
for seed in [0,1,2]:
    rng=np.random.RandomState(seed)
    for pn,cc in protos.items():
        accs=[]
        for _ in range(EP):
            Xs,ys,Xq,yq=[],[],[],[]
            ok=True
            for i,c in enumerate(cc):
                f=real[c]
                if f.shape[0]<K+1: ok=False; break
                idx=rng.permutation(f.shape[0]); qn=min(Q,f.shape[0]-K)
                Xs.append(f[idx[:K]]); ys+=[i]*K; Xq.append(f[idx[K:K+qn]]); yq+=[i]*qn
            if not ok: continue
            clf=LogisticRegression(max_iter=2000,C=1.0).fit(np.concatenate(Xs),np.array(ys))
            accs.append(clf.score(np.concatenate(Xq),np.array(yq)))
        res[pn].append(np.mean(accs)*100)
print("\n=== Linear probe (LR on frozen DINOv2, 10-shot), mean +/- std over 3 seeds ===")
for pn in protos:
    a=np.array(res[pn]); print(f"{pn:14s} {a.mean():.1f} +/- {a.std():.1f}")
