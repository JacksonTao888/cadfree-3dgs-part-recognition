# Environments

The pipeline spans four stages that do not share dependencies, so the paper's runs used
separate conda environments. Only the recognizer half is pip-installable.

| Stage | What it needs | Notes |
|---|---|---|
| Frame extraction | ffmpeg | any recent build |
| Camera poses | COLMAP + OpenCV | COLMAP is a system binary, not a pip package |
| Segmentation | SAM2 | https://github.com/facebookresearch/sam2, plus the SAM2.1 Hiera-Large checkpoint (~856 MB, not in this repo) |
| 3DGS fit and rendering | a CUDA 3D Gaussian Splatting rasterizer | built from the official gaussian-splatting / SuGaR tree; has its own non-commercial terms |
| Recognizer and figures | `requirements.txt` | torch 2.7.1+cu118, DINOv2 pulled through `torch.hub` |

Hardware used for every timing in the paper: one NVIDIA RTX 5000 Ada, 32 GB.

## Configuration

No path is hard-coded. Every script reads these environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `CADFREE_ROOT` | `~/cadfree_work` | working directory holding per-clip reconstruction folders and the assembled datasets |
| `CADFREE_VIDEOS` | `~/turntable_videos/extracted` | source videos (video A) |
| `CADFREE_TESTVID` | `~/turntable_videos_test/extracted` | test videos (video B) |
| `DTD_DIR` | `~/datasets/dtd/images` | Describable Textures Dataset, used only for the background-randomization control |
| `CADFREE_FIGOUT` | `figures/out` | where figure scripts write |
