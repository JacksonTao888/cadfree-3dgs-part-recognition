"""Extra rows for the cross-environment table (Table 1 format): no-meta 1-NN with 10 rig shots only, no-meta ProtoNet
(zero: renders prototypes; 10: rig-shot prototypes), linear probe on 10 rig shots -> home queries.  P1/P2/P3, 3 seeds."""
import os, sys, csv, json, random, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/recon_runs")); from run_meta import feats_for_dir, DEV, ROOT
allcls = sorted(r["class"] for r in csv.DictReader(open(f"{ROOT}/clip_to_class.csv")))
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).to(DEV).eval()
def cache(d): return {c: F.normalize(feats_for_dir(model, f"{d}/{c}", cap=None), dim=-1) for c in allcls}
R, B, home = cache(f"{ROOT}/data_3dgs_source"), cache(f"{ROOT}/data_real_test_rawbg"), cache(f"{ROOT}/data_home_dark")
protos = {"P1": allcls, "P2": [c for c in allcls if c.startswith("screw")], "P3": [c for c in allcls if c.startswith("nut")]}
def run(classes, rng, mode, n=300, k=10, q=15):
    accs = []
    for _ in range(n):
        pools, Q, y = [], [], []
        for i, c in enumerate(classes):
            ib = torch.randperm(B[c].shape[0], generator=rng)[:k]; ih = torch.randperm(home[c].shape[0], generator=rng)[:q]
            if mode == "nn_rig": pools.append(B[c][ib])
            elif mode == "proto_rig": pools.append(F.normalize(B[c][ib].mean(0, keepdim=True), dim=-1))
            elif mode == "proto_renders": pools.append(F.normalize(R[c].mean(0, keepdim=True), dim=-1))
            Q.append(home[c][ih]); y += [i] * ih.shape[0]
        Q = torch.cat(Q); y = torch.tensor(y)
        accs.append((torch.stack([(Q @ p.t()).max(1).values for p in pools], 1).argmax(1) == y).float().mean().item())
    return 100 * float(np.mean(accs))
def linear_probe(classes, rng, n=30, k=10):
    accs = []
    for _ in range(n):
        X, Y, Xt, Yt = [], [], [], []
        for i, c in enumerate(classes):
            ib = torch.randperm(B[c].shape[0], generator=rng)[:k]; X.append(B[c][ib]); Y += [i] * k; Xt.append(home[c]); Yt += [i] * home[c].shape[0]
        X = torch.cat(X).to(DEV); Y = torch.tensor(Y).to(DEV); Xt = torch.cat(Xt).to(DEV); Yt = torch.tensor(Yt).to(DEV)
        W = torch.zeros(X.shape[1], len(classes), device=DEV, requires_grad=True); b = torch.zeros(len(classes), device=DEV, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=1e-2)
        for _ in range(300):
            loss = F.cross_entropy(X @ W + b, Y) + 1e-3 * (W ** 2).sum(); opt.zero_grad(); loss.backward(); opt.step()
        accs.append(((Xt @ W + b).argmax(1) == Yt).float().mean().item())
    return 100 * float(np.mean(accs))
out = {}
for seed in (0, 1, 2):
    rng = torch.Generator().manual_seed(seed + 1); torch.manual_seed(seed); out[seed] = {}
    for p, cls in protos.items():
        out[seed][p] = {"nn_rig10": run(cls, rng, "nn_rig"), "proto_rig10": run(cls, rng, "proto_rig"), "proto_renders0": run(cls, rng, "proto_renders"), "linear_probe": linear_probe(cls, rng)}
        print(f"seed {seed} {p}: " + ", ".join(f"{k} {v:.1f}" for k, v in out[seed][p].items()), flush=True)
json.dump(out, open(f"{ROOT}/logs/home/crossenv_extra.json", "w"), indent=1)
print("\nmean ± sd over 3 seeds:")
for p in protos:
    print(p, "  ".join(f"{k} {np.mean([out[s][p][k] for s in out]):.1f}±{np.std([out[s][p][k] for s in out]):.1f}" for k in out[0][p]))
print("EXTRA DONE")
