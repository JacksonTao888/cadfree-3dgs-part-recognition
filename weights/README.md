# Trained recognizer

Everything needed to run the paper's recognizer on the 18 FGIP parts, with no training and no dataset.

| File | What it is | Size |
|---|---|---|
| `head_fgip18_seed0.pt` | The meta-trained head, seed 0. `recognizer/predict.py` uses it by default. | 2.1 MB |
| `head_fgip18_seed1.pt`, `head_fgip18_seed2.pt` | The same head trained with seeds 1 and 2. | 2.1 MB each |
| `gallery_fgip18_renders.npz` | The reference set: frozen-DINOv2 features of the 360 rendered views of each part, 6,480 views. | 9.2 MB |
| `heads.json` | Architecture, training settings, environment and the measured accuracy of each head. | |
| `SHA256SUMS` | Checksums of the four files above; check with `cd weights && sha256sum -c SHA256SUMS`. | |

The backbone is not stored here. It is the public DINOv2 ViT-B/14, which `torch.hub` downloads the first
time a script runs (about 350 MB).

## Run it

```bash
pip install -r requirements.txt
python recognizer/predict.py examples/*.jpg
```

`examples/` holds one test crop per part: the middle frame of each part's test video, taken without
looking at the result. A GPU is used when there is one and is not required, and the script was checked
with PyTorch 2.5.1, 2.7.1, 2.10.0 and 2.12.0. The output with the default head:

```
head:       weights/head_fgip18_seed0.pt
references: weights/gallery_fgip18_renders.npz: 18 classes, 6480 views

image                 prediction  paper name  cosine  next candidates
examples/gear_a.jpg   gear_a      gear a      0.992   nut_e 0.180  gear_b 0.117
examples/gear_b.jpg   gear_b      gear b      0.981   screw_f 0.194  nut_h 0.170
examples/nut_a.jpg    nut_a       nut a       0.928   nut_e 0.395  nut_g 0.190
examples/nut_b.jpg    nut_b       nut b       0.984   nut_e 0.323  nut_g 0.261
examples/nut_c.jpg    nut_e       nut e       0.941   nut_b 0.481  nut_c 0.428
examples/nut_d.jpg    nut_d       nut d       0.947   nut_a 0.200  screw_g 0.159
examples/nut_e.jpg    nut_e       nut e       0.978   nut_a 0.330  nut_b 0.259
examples/nut_g.jpg    nut_g       nut f       0.930   nut_d 0.512  nut_a 0.380
examples/nut_h.jpg    nut_h       nut g       0.948   nut_d 0.310  screw_f 0.285
examples/nut_i.jpg    nut_i       nut h       0.977   screw_e 0.260  screw_f 0.224
examples/screw_a.jpg  screw_a     screw a     0.976   screw_e 0.320  screw_b 0.306
examples/screw_b.jpg  screw_a     screw a     0.980   screw_b 0.416  screw_g 0.208
examples/screw_c.jpg  screw_c     screw c     0.921   screw_g 0.370  screw_f 0.309
examples/screw_d.jpg  screw_d     screw d     0.837   screw_b 0.483  screw_a 0.444
examples/screw_e.jpg  screw_e     screw e     0.984   nut_b 0.448  screw_g 0.192
examples/screw_f.jpg  screw_f     screw f     0.988   screw_b 0.325  screw_a 0.247
examples/screw_g.jpg  screw_g     screw g     0.790   screw_e 0.754  screw_a 0.424
examples/screw_h.jpg  screw_h     screw h     0.966   screw_f 0.446  screw_e 0.192
```

16 of the 18 are right; `nut_c` is taken for `nut_e` and `screw_b` is taken for `screw_a`. That is the
zero-shot setting of the paper: the reference set holds rendered views only, no real image of any part.
`cosine` is the similarity to the nearest reference view of the predicted class. `--csv out.csv` writes
the same table to a file, with the name of that nearest view.

## What the head file is

A PyTorch `state_dict` of five tensors, 525,057 values: `Linear(768, 512)`, ReLU, `Linear(512, 256)`, and
the scalar `logit_scale` that only the training loss uses. The output is L2-normalized. The class is `Head`
in `recognizer/predict.py` and `recognizer/run_meta.py`.

Each head was trained by

```bash
python recognizer/run_meta.py --cls nn --seed SEED --save-head weights/head_fgip18_seedSEED.pt
```

which is 2,000 episodes of 8 parts with 5 support and 10 query views per part, drawn from the 6,480
rendered views, Adam at 1e-3, with the 1-NN-consistent loss. No real image is used in training.

## Accuracy of these files

The paper's protocol: 300 episodes, each drawing 10 support and 15 query images per part from the test
video. In "10-shot" the reference set is those 10 real images per part and nothing else; in "zero-shot"
it is the rendered views. P1 is all 18 parts, P2 the 8 screws, P3 the 8 nuts.

| Head | P1 10-shot | P1 zero-shot | P2 10-shot | P2 zero-shot | P3 10-shot | P3 zero-shot |
|---|---|---|---|---|---|---|
| seed 0 | 91.8 | 76.1 | 90.7 | 74.2 | 90.8 | 75.5 |
| seed 1 | 91.9 | 73.3 | 91.9 | 70.4 | 91.2 | 71.7 |
| seed 2 | 92.5 | 77.4 | 92.0 | 75.6 | 91.3 | 76.5 |
| mean of the three | 92.0 | 75.6 | 91.5 | 73.4 | 91.1 | 74.6 |
| paper | 91.6 | 75.5 | 91.0 | 74.0 | 90.7 | 76.0 |

These are retrained heads, not the three behind the paper's numbers. Those were not kept, and
`run_meta.py` did not then seed the head's initial weights, so they cannot be regenerated. The files here
were trained on 2026-11-17 with the same code, data, settings and seeds; their three-seed means are within
1.5 points of the paper's in every column. The initial weights now follow `--seed` as well, and the
training command above reproduces each file byte for byte on the machine used (one RTX 5000 Ada,
torch 2.7.1+cu118). Other hardware or library versions can change the last digits.

Labelling every test image once against the rendered views (`predict.py --eval`) gives
75.97%, 72.86% and 77.11% of the 1,927 images for seeds 0, 1 and 2. The parts are not equally easy:
with seed 0, `gear_a` and `gear_b` are at 100% and `screw_b` at 7.3%; `heads.json` has every part.

## Check them against the dataset

The FGIP data is shared on request (see the main README). With it unpacked at `$F`:

```bash
F=/path/to/FGIP
python recognizer/predict.py --eval $F/test/data_real_test
python recognizer/run_meta.py --cls nn --seed 0 --load-head weights/head_fgip18_seed0.pt \
    --source $F/source/data_3dgs_source --test $F/test/data_real_test --classes-csv $F/splits/clip_to_class.csv
```

The first prints 75.97% (1464/1927) and the accuracy of each part. The second skips the
meta-training and prints the seed-0 row of the table above; for the other two rows give `--seed 1` with
the seed-1 file and `--seed 2` with the seed-2 file, because the evaluation episodes follow `--seed`.

## Your own reference images

```bash
python recognizer/predict.py --refs my_refs photo.jpg          # my_refs/<class>/*.jpg replace the rendered views
python recognizer/predict.py --add-refs my_refs photo.jpg      # ... or are added to them
python recognizer/predict.py --refs my_refs --save-gallery my_refs.npz    # store them as features once
python recognizer/predict.py --gallery my_refs.npz photo.jpg              # ... and reuse them
```

Ten real images per part as the whole reference set is the paper's 10-shot setting.

## What the images must look like

One part per image, in a square crop that the part nearly fills, on a white background. Both the rendered
views and the test images were cut out with a segmentation mask, cropped to a square around the mask's
bounding box with an 18% margin, and placed on white (`data_prep/recompose_cropped.py`). The image is then
resized to 224 pixels on its short side and centre-cropped, so a long image loses its ends. An image with
an alpha channel is placed on white first.

## Other parts

The head was meta-trained on these 18 parts. For other parts the paper's route is to render reference
views of the new parts and run the meta-training on them, which takes seconds once the features are
computed (`run_meta.py --source ... --save-head ...`). `predict.py --refs` also runs the released head on
any set of classes, but the paper did not measure how it does on parts it was not trained on.

## Names

The data has no `nut_f`: its `nut_g`, `nut_h` and `nut_i` are the paper's nut f, nut g and nut h.
`predict.py` prints both names.
