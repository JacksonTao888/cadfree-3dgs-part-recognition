# Reproducing the paper

This file maps each table and figure to the code that produced it, and is explicit about what cannot
be reproduced from this repository alone.

## The honest summary

The FGIP data is not published here (about 19 GB of videos, frames, masks and twins), so the paper's
exact numbers cannot be regenerated from a fresh clone. What you can do is run the whole pipeline on
your own turntable capture, and inspect the small result files under `results/` that the paper's
tables were read from.

## Table and figure map

| Paper item | Script | Notes |
|---|---|---|
| Table 1, source comparison on the rig | `recognizer/run_meta.py`, driven by `scripts/run_experiments.sh` | the 3-seed headline (91.6 / 75.5) was aggregated outside any committed driver; see gaps |
| Table 2, cross-environment | `recognizer/eval_crossenv_final.py` | per-seed outputs in `results/crossenv/` |
| Table 3, robustness and backbones | `recognizer/baseline_extra.py`, `recognizer/clip_meta.py` | |
| Table 4, reconstruction fidelity | `analysis/render_quality.py` | output in `results/render_quality.csv` |
| Table 5, computational cost | `analysis/bench_inference.py`, `scripts/time_pipeline.sh` | output in `results/bench_inference.json` |
| Figure 1, overview | none | hand-authored in a vector tool; there is no producer script |
| Figure 2, recognizer schematic | `figures/fig_recognizer.py` | |
| Figure 3, dataset grid | `figures/fig_dataset.py` | |
| Figure 4, results bars | `figures/fig_results_bars.py` | see gaps: values are transcribed literals |
| Figure 5, elevation strip | `figures/fig_elevation.py` | see gaps: its input directory has no committed producer |
| Figure 6, render fidelity | `figures/fig_fidelity.py` | |
| Figure 7, view redundancy | `analysis/symmetry_index.py`, `results/symmetry_index.csv` | |
| Figures 8–9, confusion matrices | `analysis/confusion_episodic.py` then `figures/fig_confusion_p1.py` / `fig_confusion_p23.py` | matrices in `results/confusion/` |
| Figure 10, per-part accuracy | `figures/fig_results_bars.py` | reads `results/confusion/` |
| Figure 11, t-SNE | `figures/fig_tsne.py` | |

## Known gaps

These are stated rather than papered over.

1. **The 3-seed headline has no committed driver.** 91.6 ± 1.4 / 75.5 ± 3.1 was aggregated by hand
   across three seed runs; `scripts/run_experiments.sh` reproduces the individual runs but not the
   aggregation step.
2. **The matched-120 datasets have no builder.** `data_3dgs_base120` and `data_rawframes_m120`, which
   produce the matched-view comparison in Table 1 and the reversal in Table 2, were assembled ad hoc.
   Only readers of those directories are committed.
3. **Figure 4 hard-codes its numbers.** The values are Python literals transcribed from the paper
   rather than read from `results/`, so re-running an experiment will not update the figure.
4. **Figure 5's input has no producer.** The elevation sweep directory came from a one-off edit of the
   orbit elevations in `reconstruction/render_final.py`.
5. **The Section 4.6 gradient-magnitude analysis has no committed code.** The render/real
   mean-gradient-magnitude ratios and their correlation with accuracy were computed ad hoc and the
   script did not survive; the resulting numbers are in `results/fidelity_vs_accuracy.json`.
6. **`results/fidelity_vs_accuracy.json` is partly superseded.** Its stored correlation values predate
   the final runs and do not match the manuscript; use `results/render_quality.csv` joined to
   `results/confusion/` instead.

## Part naming

The eighteen parts are displayed as `gear a-b`, `nut a-h`, `screw a-h`. Three nuts carry different
internal identifiers: display `nut f`, `nut g`, `nut h` correspond to internal `nut_g`, `nut_h`,
`nut_i`. The figure scripts apply this mapping.
