import os
import numpy as np, torch, csv, random
from run_matrix import load_backbone, feats_for_dir, episode_acc, ROOT
SRC=os.environ.get("SRC",f"{ROOT}/data_3dgs_source"); CAP=int(os.environ.get("CAP","400")) or None
allcls=sorted(r["class"] for r in csv.DictReader(open(f"{ROOT}/clip_to_class.csv")))
model=load_backbone("dinov2_vitb14")
print("caching features ...", flush=True)
src={c:feats_for_dir(model,f"{SRC}/{c}",cap=CAP) for c in allcls}
real={c:feats_for_dir(model,f"{ROOT}/data_real_test/{c}") for c in allcls}
protos={"P1":allcls,"P2":[c for c in allcls if c.startswith("screw")],"P3":[c for c in allcls if c.startswith("nut")]}
res={}
for seed in [0,1,2]:
    rng=torch.Generator().manual_seed(seed); random.seed(seed)
    for name,cls in protos.items():
        for reg,lbl in [("support-only","10shot"),("source-only","zeroshot")]:
            acc,ci=episode_acc(src,real,cls,reg,10,15,300,rng)
            res.setdefault((name,lbl),[]).append(acc)
print("\n=== baseline (no-meta ProtoNet, raw frozen DINOv2) 3-seed mean +/- std ===", flush=True)
for name in ["P1","P2","P3"]:
    for lbl in ["10shot","zeroshot"]:
        v=np.array(res[(name,lbl)])
        print(f"  {name} {lbl:8s}: {v.mean():.1f} +/- {v.std():.1f}   seeds={[round(x,1) for x in res[(name,lbl)]]}", flush=True)
print("DONE_BASELINE_SEEDS", flush=True)
