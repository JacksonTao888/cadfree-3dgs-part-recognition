"""
Meta-learning over frozen DINOv2 features (the method; ProtoNet baseline = run_matrix.py).
A small projection head is META-TRAINED episodically on the 3DGS SOURCE domain
(learn-to-form-good-prototypes), then used to embed real images at test time.

Compares, per protocol, the same 3 regimes as run_matrix but in the META-LEARNED
embedding: support-only / source-only(0-shot) / source+fewshot. If the meta head
beats raw-feature ProtoNet, meta-learning earns its place (and should fix the
naive source+fewshot drop).
"""
import os
import argparse, os, glob, csv, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torchvision import transforms

torch.hub.set_dir(os.path.expanduser("~/.cache/torch/hub"))
ROOT = os.environ.get("CADFREE_ROOT", os.path.expanduser("~/cadfree_work")); DEV = "cuda" if torch.cuda.is_available() else "cpu"
TFM = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

@torch.no_grad()
def feats_for_dir(model, d, variant="dinov2_vitb14", batch=64, cap=400):
    paths = sorted(glob.glob(f"{d}/*.jpg") + glob.glob(f"{d}/*.png"))
    if cap and len(paths) > cap:
        random.Random(0).shuffle(paths); paths = paths[:cap]
    # optional on-disk feature cache (FEAT_CACHE=1): keyed by variant + resolved file list + mtimes; default off (paper behaviour unchanged)
    ck = None
    if os.environ.get("FEAT_CACHE") == "1" and paths:
        import hashlib
        key = hashlib.sha1(("\n".join(f"{os.path.realpath(p)}|{os.path.getmtime(p):.0f}" for p in paths) + f"|{variant}").encode()).hexdigest()
        os.makedirs(f"{ROOT}/_featcache", exist_ok=True); ck = f"{ROOT}/_featcache/{key}.pt"
        if os.path.exists(ck):
            return torch.load(ck)
    out = []
    for i in range(0, len(paths), batch):
        ims = torch.stack([TFM(Image.open(p).convert("RGB")) for p in paths[i:i+batch]]).to(DEV)
        f = model.forward_features(ims)["x_norm_clstoken"] if variant.startswith("dinov2") else model(ims)
        out.append(F.normalize(f, dim=-1).cpu())
    feats = torch.cat(out) if out else torch.zeros(0, 768)
    if ck: torch.save(feats, ck)
    return feats

class Head(nn.Module):
    def __init__(self, d=768, h=512, o=256):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d,h), nn.ReLU(), nn.Linear(h,o))
        self.logit_scale = nn.Parameter(torch.tensor(10.0))
    def forward(self, x): return F.normalize(self.net(x), dim=-1)

def sample_episode(feats, classes, n_way, k_shot, q, rng):
    cs = random.sample(classes, n_way)
    sup, qry, qy = [], [], []
    for i, c in enumerate(cs):
        f = feats[c]; idx = torch.randperm(f.shape[0], generator=rng)[:k_shot+q]
        sup.append(f[idx[:k_shot]]); qry.append(f[idx[k_shot:k_shot+q]]); qy += [i]*(len(idx)-k_shot)
    return torch.cat(sup).to(DEV), [s.shape[0] for s in sup], torch.cat(qry).to(DEV), torch.tensor(qy).to(DEV), n_way

def protos_from(emb, counts, n_way):
    out=[]; o=0
    for i in range(n_way):
        out.append(F.normalize(emb[o:o+counts[i]].mean(0), dim=-1)); o+=counts[i]
    return torch.stack(out)

def meta_train(head, src, classes, iters, n_way, k_shot, q, lr, seed, cls_mode="proto"):
    opt = torch.optim.Adam(head.parameters(), lr=lr); rng = torch.Generator().manual_seed(seed)
    head.train()
    for t in range(iters):
        nw = min(n_way, len(classes))
        sup, cnt, qry, qy, nw = sample_episode(src, classes, nw, k_shot, q, rng)
        Se = head(sup); Q = head(qry)
        if cls_mode == "nn":   # train/test-consistent soft-1-NN: logsumexp over class views
            sims = head.logit_scale * (Q @ Se.t()); o = 0; cols = []
            for i in range(nw):
                cols.append(torch.logsumexp(sims[:, o:o+cnt[i]], dim=1)); o += cnt[i]
            logits = torch.stack(cols, dim=1)
        else:                  # mean-prototype
            P = protos_from(Se, cnt, nw); logits = head.logit_scale * (Q @ P.t())
        loss = F.cross_entropy(logits, qy)
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()

@torch.no_grad()
def eval_regime(head, src, real, classes, regime, k_shot, k_query, episodes, rng, cls_mode="proto"):
    accs=[]
    for _ in range(episodes):
        pools=[]; qf=[]; qy=[]; ok=True
        for i,c in enumerate(classes):
            rf=real[c]
            if rf.shape[0] < k_shot+1: ok=False; break
            idx=torch.randperm(rf.shape[0], generator=rng)
            sup=rf[idx[:k_shot]].to(DEV); qn=min(k_query, rf.shape[0]-k_shot); qry=rf[idx[k_shot:k_shot+qn]].to(DEV)
            if regime=="support-only": e=head(sup)
            elif regime=="source-only": e=head(src[c].to(DEV))
            else: e=head(torch.cat([src[c].to(DEV), sup]))
            pools.append(e)                       # per-class embeddings (normalized by head)
            qf.append(head(qry)); qy += [i]*qry.shape[0]
        if not ok: continue
        Q=torch.cat(qf); y=torch.tensor(qy).to(DEV)
        if cls_mode=="nn":                        # query -> max cosine to ANY view in the class
            scores=torch.stack([(Q @ p.t()).max(1).values for p in pools], dim=1)
        else:                                     # mean prototype
            P=torch.stack([F.normalize(p.mean(0), dim=-1) for p in pools]); scores=Q @ P.t()
        accs.append((scores.argmax(1)==y).float().mean().item())
    a=np.array(accs); return (a.mean()*100, 1.96*a.std()/np.sqrt(len(a))*100) if len(a) else (float('nan'),)*2

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("--source", default=f"{ROOT}/data_3dgs_source")
    pa.add_argument("--test", default=f"{ROOT}/data_real_test")
    pa.add_argument("--variant", default="dinov2_vitb14")
    pa.add_argument("--meta-iters", type=int, default=2000)
    pa.add_argument("--meta-nway", type=int, default=8)
    pa.add_argument("--shots", type=int, default=10); pa.add_argument("--k-query", type=int, default=15)
    pa.add_argument("--episodes", type=int, default=300); pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--cls", default="proto", choices=["proto","nn"], help="classifier: mean prototype or 1-NN over views")
    pa.add_argument("--classes-csv", default=f"{ROOT}/clip_to_class.csv", help="restrict to classes in this csv (pilot subset)")
    pa.add_argument("--cap", type=int, default=400, help="max images cached per class dir (0 = no cap)")
    pa.add_argument("--gallery", default="", help="optional dir used as the zero-shot reference gallery instead of --source (meta-training still uses --source)")
    pa.add_argument("--save-head", default="", help="write the meta-trained head (a state_dict) to this file")
    pa.add_argument("--load-head", default="", help="skip meta-training and evaluate this saved head instead (e.g. weights/head_fgip18_seed0.pt)")
    args=pa.parse_args(); random.seed(args.seed)
    allcls=sorted(r["class"] for r in csv.DictReader(open(args.classes_csv)))
    if args.variant.startswith("dinov2"):
        model=torch.hub.load("facebookresearch/dinov2", args.variant, trust_repo=True).to(DEV).eval()
        fdim={"dinov2_vits14":384,"dinov2_vitb14":768,"dinov2_vitl14":1024}.get(args.variant,768)
    else:
        from torchvision import models as tvm
        model=tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1); model.fc=nn.Identity(); model=model.to(DEV).eval(); fdim=512
    print(f"caching features ({args.variant}, dim={fdim}) ...")
    cap=args.cap if args.cap>0 else None
    src={c:feats_for_dir(model,f"{args.source}/{c}",args.variant,cap=cap) for c in allcls}
    real={c:feats_for_dir(model,f"{args.test}/{c}",args.variant,cap=cap) for c in allcls}
    gal={c:feats_for_dir(model,f"{args.gallery}/{c}",args.variant,cap=cap) for c in allcls} if args.gallery else src
    torch.manual_seed(args.seed)   # the head's initial weights follow --seed too, so a run can be repeated exactly
    head=Head(d=fdim).to(DEV)
    if args.load_head:
        head.load_state_dict(torch.load(args.load_head, map_location=DEV, weights_only=True)); head.eval()
        print(f"loaded head {args.load_head} (no meta-training)")
    else:
        print(f"meta-training head on SOURCE ({args.meta_iters} iters, {args.meta_nway}-way) ...")
        meta_train(head, src, allcls, args.meta_iters, args.meta_nway, 5, 10, 1e-3, args.seed, args.cls)
        if args.save_head:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_head)), exist_ok=True)
            torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, args.save_head)
            print(f"saved head -> {args.save_head}")
    rng=torch.Generator().manual_seed(args.seed+1)
    protos={"P1_all(18way)":allcls,"P2_screw(8way)":[c for c in allcls if c.startswith("screw")],"P3_nut(8way)":[c for c in allcls if c.startswith("nut")]}
    print(f"\nMETA over DINOv2 | source={os.path.basename(args.source)} gallery={os.path.basename(args.gallery) if args.gallery else os.path.basename(args.source)} test={os.path.basename(args.test)} shots={args.shots}")
    print(f"{'protocol':16s} {'support-only':>14s} {'source-only':>14s} {'source+fewshot':>16s}")
    for name,cls in protos.items():
        line=f"{name:16s}"
        for reg in ["support-only","source-only","source+fewshot"]:
            acc,ci=eval_regime(head,gal,real,cls,reg,args.shots,args.k_query,args.episodes,rng,args.cls)
            line+=f" {acc:6.2f}±{ci:4.2f}  "
        print(line)

if __name__=="__main__": main()
