"""Computational-cost benchmark of the recognizer (R2 comment 6).
Measures, on the current GPU: parameter counts, FLOPs per image (DINOv2 ViT-B/14 @224 + head),
feature-extraction throughput/latency (batch 1 and 64), peak GPU memory, meta-training wall time for
2000 episodes on cached features, and 1-NN query latency against an 18x360-view gallery.
Run with the skill-meta env:  ~/miniconda3/envs/skill-meta/bin/python bench_inference.py
"""
import os, time, json, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
torch.hub.set_dir(os.path.expanduser("~/.cache/torch/hub"))
DEV = "cuda"; ROOT = os.path.expanduser("~/recon_runs")
from run_meta import Head, meta_train

res = {"gpu": torch.cuda.get_device_name(0)}
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).to(DEV).eval()
head = Head().to(DEV)
res["backbone_params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
res["head_params_K"] = round(sum(p.numel() for p in head.parameters()) / 1e3, 1)

x1 = torch.randn(1, 3, 224, 224, device=DEV)
with torch.no_grad():
    fc = FlopCounterMode(display=False)
    with fc: model(x1)
    res["backbone_GFLOPs_per_image"] = round(fc.get_total_flops() / 1e9, 2)
    fc = FlopCounterMode(display=False)
    with fc: head(torch.randn(1, 768, device=DEV))
    res["head_MFLOPs_per_image"] = round(fc.get_total_flops() / 1e6, 3)

def bench(bs, iters=50):
    x = torch.randn(bs, 3, 224, 224, device=DEV)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(10): model(x)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(iters): head(model(x))
        torch.cuda.synchronize(); dt = (time.perf_counter() - t) / iters
    return dt, torch.cuda.max_memory_allocated() / 2**20

for bs in (1, 64):
    dt, mem = bench(bs)
    res[f"latency_ms_batch{bs}"] = round(dt * 1e3, 2)
    res[f"throughput_img_per_s_batch{bs}"] = round(bs / dt, 1)
    res[f"peak_mem_MiB_batch{bs}"] = round(mem, 0)

# meta-training wall time on cached features (synthetic feature bank of the paper's size: 18 x 360 x 768)
src = {f"c{i}": torch.randn(360, 768) for i in range(18)}
torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.perf_counter()
meta_train(head, src, sorted(src), 2000, 8, 5, 10, 1e-3, 0, "nn")
torch.cuda.synchronize(); res["meta_train_2000it_s"] = round(time.perf_counter() - t, 1)
res["meta_train_peak_mem_MiB"] = round(torch.cuda.max_memory_allocated() / 2**20, 0)

# 1-NN query against gallery of 18 classes x 360 views (embedded)
with torch.no_grad():
    gal = F.normalize(torch.randn(18 * 360, 256, device=DEV), dim=-1); q = F.normalize(torch.randn(1, 256, device=DEV), dim=-1)
    for _ in range(10): (q @ gal.t()).view(18, 360).max(1)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(200): (q @ gal.t()).view(18, 360).max(1).values.argmax()
    torch.cuda.synchronize(); res["nn_query_ms_gallery_6480"] = round((time.perf_counter() - t) / 200 * 1e3, 3)

print(json.dumps(res, indent=1))
json.dump(res, open(f"{ROOT}/logs/bench_inference.json", "w"), indent=1)
