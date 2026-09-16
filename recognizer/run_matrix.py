"""
Frozen-DINOv2 recognition matrix for the CAD-free 3DGS-source study.

Regimes (per protocol, episodic, acc +- 95% CI):
  support-only       : class prototypes from K real support shots (no source)
  source-only        : class prototypes from the 3DGS source only (ZERO real labels)
  source+fewshot     : prototypes from source + K real support shots

Protocols: all (18-way) / screw (8-way) / nut (8-way).
Features: frozen DINOv2 ViT-B/14 CLS, cached once per image. Cosine prototypes.
"""
import os
import argparse, os, glob, csv, random
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms

torch.hub.set_dir(os.path.expanduser("~/.cache/torch/hub"))
ROOT = os.environ.get("CADFREE_ROOT", os.path.expanduser("~/cadfree_work"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TFM = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])


def load_backbone(variant):
    m = torch.hub.load("facebookresearch/dinov2", variant, trust_repo=True).to(DEV).eval()
    return m

@torch.no_grad()
def feats_for_dir(model, d, batch=64, cap=400):
    paths = sorted(glob.glob(f"{d}/*.jpg") + glob.glob(f"{d}/*.png"))
    if cap and len(paths) > cap:
        random.Random(0).shuffle(paths); paths = paths[:cap]
    out = []
    for i in range(0, len(paths), batch):
        ims = torch.stack([TFM(Image.open(p).convert("RGB")) for p in paths[i:i+batch]]).to(DEV)
        f = model.forward_features(ims)["x_norm_clstoken"]
        out.append(F.normalize(f, dim=-1).cpu())
    return torch.cat(out) if out else torch.zeros(0, 768)


def proto(feats):
    return F.normalize(feats.mean(0, keepdim=True), dim=-1)[0]


def episode_acc(src, real, classes, regime, k_shot, k_query, episodes, rng):
    accs = []
    for _ in range(episodes):
        protos, q_feats, q_labels = [], [], []
        ok = True
        for i, c in enumerate(classes):
            rf = real[c]
            if rf.shape[0] < k_shot + 1: ok = False; break
            idx = torch.randperm(rf.shape[0], generator=rng)
            sup = rf[idx[:k_shot]]
            qn = min(k_query, rf.shape[0]-k_shot)
            qry = rf[idx[k_shot:k_shot+qn]]
            if regime == "support-only":
                protos.append(proto(sup))
            elif regime == "source-only":
                protos.append(proto(src[c]))
            else:  # source+fewshot
                protos.append(proto(torch.cat([src[c], sup])))
            q_feats.append(qry); q_labels += [i]*qry.shape[0]
        if not ok: continue
        P = torch.stack(protos); Q = torch.cat(q_feats); y = torch.tensor(q_labels)
        pred = (Q @ P.t()).argmax(1)
        accs.append((pred == y).float().mean().item())
    a = np.array(accs)
    return (a.mean()*100, 1.96*a.std()/np.sqrt(len(a))*100) if len(a) else (float("nan"), float("nan"))


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--source", default=f"{ROOT}/data_3dgs_source")
    pa.add_argument("--test", default=f"{ROOT}/data_real_test")
    pa.add_argument("--variant", default="dinov2_vitb14")
    pa.add_argument("--shots", type=int, default=10)
    pa.add_argument("--k-query", type=int, default=15)
    pa.add_argument("--episodes", type=int, default=300)
    pa.add_argument("--seed", type=int, default=0)
    args = pa.parse_args()
    random.seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)

    allcls = sorted(r["class"] for r in csv.DictReader(open(f"{ROOT}/clip_to_class.csv")))
    model = load_backbone(args.variant)
    print(f"caching features ({args.variant}) ...")
    src = {c: feats_for_dir(model, f"{args.source}/{c}") for c in allcls}
    real = {c: feats_for_dir(model, f"{args.test}/{c}") for c in allcls}
    for c in allcls:
        print(f"  {c:8s} source={src[c].shape[0]:4d}  real={real[c].shape[0]:4d}")

    protos = {"P1_all(18way)": allcls,
              "P2_screw(8way)": [c for c in allcls if c.startswith("screw")],
              "P3_nut(8way)":   [c for c in allcls if c.startswith("nut")]}
    print(f"\nsource={os.path.basename(args.source)}  test={os.path.basename(args.test)}  "
          f"shots={args.shots} episodes={args.episodes}")
    print(f"{'protocol':16s} {'support-only':>16s} {'source-only(0shot)':>20s} {'source+fewshot':>18s}")
    for name, cls in protos.items():
        line = f"{name:16s}"
        for reg in ["support-only", "source-only", "source+fewshot"]:
            acc, ci = episode_acc(src, real, cls, reg, args.shots, args.k_query, args.episodes, rng)
            line += f" {acc:6.2f}±{ci:4.2f}    "
        print(line)


if __name__ == "__main__":
    main()
