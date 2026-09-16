"""FINAL cross-environment evaluation (option B): sources renders / frames / combined (unchanged, video A);
tests = home dark set, white cut-outs (paper protocol) and background-kept crops.
Regimes: zero-shot (gallery = source, queries = home); 10-shot with support from VIDEO B (same-rig labels; white
cut-outs for the cut-out test, rawbg for the bg test), queries = home; B-support-only; no-meta zero-shot.
P1/P2/P3, 300 episodes, k_query 15.  usage: FEAT_CACHE=1 python eval_crossenv_final.py <seed>"""
import os, sys, csv, json, random, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/recon_runs"))
from run_meta import feats_for_dir, Head, meta_train, DEV, ROOT
seed = int(sys.argv[1]); random.seed(seed); torch.manual_seed(seed)
allcls = sorted(r["class"] for r in csv.DictReader(open(f"{ROOT}/clip_to_class.csv")))
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).to(DEV).eval()
def cache(d): return {c: feats_for_dir(model, f"{d}/{c}", cap=None) for c in allcls}
R, Fm = cache(f"{ROOT}/data_3dgs_source"), cache(f"{ROOT}/data_rawframes_source")
SOURCES = {"frames-m120": cache(f"{ROOT}/data_rawframes_m120"), "renders-base120": cache(f"{ROOT}/data_3dgs_base120")} if os.environ.get("MATCHED") == "1" else {"renders": R, "frames": Fm, "combined": {c: torch.cat([R[c], Fm[c]]) for c in allcls}}
TESTS = {"bg-kept": (cache(f"{ROOT}/data_home_dark"), cache(f"{ROOT}/data_real_test_rawbg"))}
protos = {"P1": allcls, "P2": [c for c in allcls if c.startswith("screw")], "P3": [c for c in allcls if c.startswith("nut")]}
class Ident(torch.nn.Module):
    def forward(self, x): return F.normalize(x, dim=-1)
ident = Ident().to(DEV)
@torch.no_grad()
def episodes(head, src, home, B, classes, regime, rng, n=300, k=10, q=15):
    accs = []
    for _ in range(n):
        pools, Q, y = [], [], []
        for i, c in enumerate(classes):
            ib = torch.randperm(B[c].shape[0], generator=rng)[:k]; ih = torch.randperm(home[c].shape[0], generator=rng)[:q]; sup = B[c][ib]
            e = head(sup.to(DEV)) if regime == "B-only" else (head(src[c].to(DEV)) if regime == "zero" else head(torch.cat([src[c], sup]).to(DEV)))
            pools.append(e); Q.append(head(home[c][ih].to(DEV))); y += [i] * ih.shape[0]
        Q = torch.cat(Q); y = torch.tensor(y).to(DEV)
        accs.append((torch.stack([(Q @ p.t()).max(1).values for p in pools], 1).argmax(1) == y).float().mean().item())
    return 100 * float(np.mean(accs))
res = []
for sname, src in SOURCES.items():
    head = Head().to(DEV); meta_train(head, src, allcls, 2000, 8, 5, 10, 1e-3, seed, "nn")
    for tname, (home, B) in TESTS.items():
        rng = torch.Generator().manual_seed(seed + 1); row = {"seed": seed, "source": sname, "test": tname}
        for p, cls in protos.items():
            row[p] = {"zero": episodes(head, src, home, B, cls, "zero", rng), "ten_B": episodes(head, src, home, B, cls, "ten", rng),
                      "B_only": episodes(head, src, home, B, cls, "B-only", rng), "nometa_zero": episodes(ident, src, home, B, cls, "zero", rng)}
        res.append(row)
        print(f"seed {seed} {sname:9s} {tname:8s} " + "  ".join(f"{p} zero {row[p]['zero']:5.1f} 10(B) {row[p]['ten_B']:5.1f} Bonly {row[p]['B_only']:5.1f} nm {row[p]['nometa_zero']:5.1f}" for p in protos), flush=True)
json.dump(res, open(f"{ROOT}/logs/home/crossenv_final{'_matched' if os.environ.get('MATCHED') == '1' else ''}_seed{seed}.json", "w"), indent=1); print(f"CE DONE seed {seed}")
