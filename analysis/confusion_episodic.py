"""Confusion matrices ACCUMULATED over the episodic protocol of Table 1 (renders-only source, 1-NN meta-head,
3 seeds x 300 episodes, 10 support / 15 query per class), so that their overall accuracy equals the table's mean by
construction.  Saves cm_P{1,2,3}_{10shot,zeroshot}.npy (counts, class order gear/nut/screw) and per-seed accuracies."""
import os, sys, csv, json, random, numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/recon_runs")); from run_meta import feats_for_dir, Head, meta_train, DEV, ROOT
allcls = sorted(r["class"] for r in csv.DictReader(open(f"{ROOT}/clip_to_class.csv")))
order = sorted(allcls, key=lambda c: ({"gear": 0, "nut": 1, "screw": 2}[c.split("_")[0]], c))
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).to(DEV).eval()
src = {c: feats_for_dir(model, f"{ROOT}/data_3dgs_source/{c}", cap=None) for c in allcls}
real = {c: feats_for_dir(model, f"{ROOT}/data_real_test/{c}", cap=None) for c in allcls}
P = {"P1": order, "P2": [c for c in order if c.startswith("screw")], "P3": [c for c in order if c.startswith("nut")]}
CM = {(p, r): np.zeros((len(cls), len(cls))) for p, cls in P.items() for r in ("10shot", "zeroshot")}; acc = {}
for seed in (0, 1, 2):
    random.seed(seed); torch.manual_seed(seed)
    head = Head().to(DEV); meta_train(head, src, allcls, 2000, 8, 5, 10, 1e-3, seed, "nn"); rng = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        srcE = {c: head(src[c].to(DEV)) for c in allcls}
        for p, cls in P.items():
            for reg in ("10shot", "zeroshot"):
                hits = []
                for _ in range(300):
                    pools, Q, y = [], [], []
                    for i, c in enumerate(cls):
                        rf = real[c]; idx = torch.randperm(rf.shape[0], generator=rng); sup = rf[idx[:10]].to(DEV); qry = rf[idx[10:25]].to(DEV)
                        pools.append(srcE[c] if reg == "zeroshot" else torch.cat([srcE[c], head(sup)])); Q.append(head(qry)); y += [i] * qry.shape[0]
                    Qe = torch.cat(Q); pred = torch.stack([(Qe @ pl.t()).max(1).values for pl in pools], 1).argmax(1).cpu().numpy(); y = np.array(y)
                    for t, pr in zip(y, pred): CM[(p, reg)][t, pr] += 1
                    hits.append((pred == y).mean())
                acc[(seed, p, reg)] = 100 * float(np.mean(hits)); print(f"seed {seed} {p} {reg}: {acc[(seed, p, reg)]:.1f}", flush=True)
for (p, reg), M in CM.items():
    np.save(f"{ROOT}/cm_{p}_{reg}.npy", M); print(f"{p} {reg}: accumulated accuracy {100 * np.trace(M) / M.sum():.1f}  (3-seed mean of episode means {np.mean([acc[(s, p, reg)] for s in (0, 1, 2)]):.1f} ± {np.std([acc[(s, p, reg)] for s in (0, 1, 2)]):.1f})")
json.dump({f"{s}|{p}|{r}": v for (s, p, r), v in acc.items()}, open(f"{ROOT}/logs/confusion_episodic_acc.json", "w"), indent=1); print("CMEP DONE")
