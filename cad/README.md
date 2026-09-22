# FGIP parts: GB standards, AutoCAD locations, and rebuilt STL models

The 18 parts of the FGIP set were printed from the GB standard-part library built into
AutoCAD Mechanical, enlarged. This folder records which standard each part is, where to find it
in AutoCAD, and a parametric STL rebuild of each one.

The STL files are **reconstructions to the standard's nominal dimension table**, not the original
CAD. Threads are not modelled, which matches the simplified representation AutoCAD inserts and
which the photographs show. Each part was identified from scale-free proportions measured on its
3D Gaussian Splatting twin and on the video-B photographs; the nominal size was then chosen so
that those proportions match. `gen_gb_parts.py` regenerates everything, one named argument per
dimension.

## Where each part lives in AutoCAD Mechanical

Insert from the standard-parts browser: **内容 (Content) 选项卡 → 标准零件 → 紧固件**, or open the
content library browser and search for the GB number directly, which is the reliable route because
the menu wording differs between versions and language packs. The two gears are not standard parts;
they come from the cylindrical gear generator under **内容 → 计算 → 齿轮**.

| dataset | paper | standard | AutoCAD path |
|---|---|---|---|
| screw_a | screw_a | GB/T 70.1 内六角圆柱头螺钉 | 紧固件 → 螺钉 → 内六角螺钉 → GB/T 70.1 |
| screw_d | screw_d | GB/T 5783 六角头螺栓 全螺纹 | 紧固件 → 螺栓 → 六角头螺栓 → GB/T 5783 |
| screw_f | screw_f | GB/T 5281 内六角圆柱头轴肩螺钉 | 紧固件 → 螺钉 → 轴肩螺钉 → GB/T 5281 |
| screw_g | screw_g | GB/T 65 开槽圆柱头螺钉 | 紧固件 → 螺钉 → 开槽螺钉 → GB/T 65 |
| nut_a | nut_a | GB/T 6170 1型六角螺母 | 紧固件 → 螺母 → 六角螺母 → GB/T 6170 |
| nut_b | nut_b | GB/T 6178 1型六角开槽螺母 | 紧固件 → 螺母 → 开槽螺母 → GB/T 6178 |
| nut_d | nut_d | GB/T 812 圆螺母 | 紧固件 → 螺母 → 圆螺母 → GB/T 812 |
| nut_e | nut_e | GB/T 6175 2型六角螺母 | 紧固件 → 螺母 → 六角螺母 → GB/T 6175 |
| nut_g | nut_f | GB/T 6172.1 六角薄螺母 | 紧固件 → 螺母 → 六角螺母 → GB/T 6172.1 |
| nut_h | nut_g | GB/T 6177.1 六角法兰面螺母 | 紧固件 → 螺母 → 法兰螺母 → GB/T 6177.1 |
| nut_i | nut_h | GB/T 56 六角厚螺母 | 紧固件 → 螺母 → 六角螺母 → GB/T 56 |
| gear_a | gear_a | 直齿圆柱齿轮, z = 20 (齿廓 GB/T 1356, 模数 GB/T 1357) | 内容 → 计算 → 齿轮 → 圆柱齿轮 |
| gear_b | gear_b | 斜齿圆柱齿轮, z = 13 | 内容 → 计算 → 齿轮 → 圆柱齿轮 (螺旋角 ≠ 0) |

Evidence for each identification, as measured ratio versus the standard's value:

| part | ratio | measured | standard |
|---|---|---|---|
| screw_a | head dia / shank dia | 1.478 | 1.50 |
| screw_a | head height / shank dia | 1.027 | 1.00 |
| screw_d | head height / shank dia | 0.701 | 0.64 |
| screw_f | head dia / shoulder dia | 1.430 | 1.50 |
| screw_f | thread dia / shoulder dia | 0.81 | 0.80 |
| screw_g | head dia / shank dia | 1.543 | 1.60 |
| nut_a | height / across flats | 0.63 | 0.60 |
| nut_b | height / across flats | 0.80 | 0.79 |
| nut_d | outer dia / height | 3.49 | 3.50 |
| nut_e | height / across flats | 0.68 | 0.68 |
| nut_g (f) | height / across flats | 0.30 | 0.31 |
| nut_h (g) | height / across flats | 0.81 | 0.81 |
| nut_i (h) | height / across flats | 0.81 | 0.79 |
| gear_a | tooth count | 20 | — |
| gear_b | tooth count | 13 | — |

## Not settled

| dataset | geometry | why it is open |
|---|---|---|
| nut_c | plain hex nut, height / across flats = 0.96 | thicker than 六角厚螺母 GB/T 56 (0.71–0.83); `nut_c__thick_hex_TBD.stl` is a placeholder at the measured ratio |
| screw_b | hex-socket cylindrical head, head dia / shank 1.44, head height / shank 0.76 | GB/T 70.1 has head height / shank = 1.00 at every size, so this is a low-head variant or another standard |
| screw_c | hex head, then a thinner section, then a thicker chamfered cylindrical end | that order matches neither a partly threaded bolt nor a fitted bolt |
| screw_e | tall cylindrical head, slot on the top face, transverse through-hole in the head | a wire-lock or tommy-bar hole; 滚花高头螺钉 GB/T 834 is the leading candidate |
| screw_h | slotted head, length / shank dia ≈ 9, a boss at the far end | its twin is the weakest in the set (19.9 dB), so the measurement is unreliable |

## A note on absolute size

The ratios identify the standard; they do not identify the nominal size on their own, and the size
chosen here is the one whose ratios fit best. For screw_a the ratio head dia / shank dia = 1.48
only occurs at M12 and above in GB/T 70.1 (it is 1.83 at M3 and 1.60 at M10), so the nominal source
was not a small screw. If the printed parts really are 10× the nominal, screw_a would be 570 mm
long; measure one and, if that disagrees, rerun `gen_gb_parts.py` with the size you actually used.

## Contents

- `stl_nominal/` — each part at its GB nominal size, in millimetres
- `stl_10x/` — the same models scaled ten times
- `gen_gb_parts.py` — the generator, one named argument per dimension
- `verify.py` — compares a generated model's silhouette with the proportions measured from the scans
- `preview_real_vs_generated.jpg` — each generated part beside a photograph of the real one
