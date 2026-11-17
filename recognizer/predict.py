"""
Recognise a part from an image with the released head. No training and no dataset are needed.

    python recognizer/predict.py examples/*.jpg              # label each image
    python recognizer/predict.py --eval DIR                  # DIR/<class>/*.jpg -> accuracy
    python recognizer/predict.py --refs MY_REFS photo.jpg    # use your own reference images

This is the paper's test-time rule and nothing else: a frozen DINOv2 ViT-B/14 turns each 224x224
image into a 768-d feature, the meta-trained head maps it to a 256-d unit vector, and the image takes
the label of its most similar reference view (1-NN on cosine similarity).

The default reference set is weights/gallery_fgip18_renders.npz, the DINOv2 features of the 360
rendered views of each of the 18 FGIP parts. That is the paper's zero-shot setting: no real image of
any part is in the reference set. --refs replaces it with a folder of your own images laid out as
DIR/<class>/*.jpg (the paper's 10-shot setting when the folder holds 10 real images per class), and
--add-refs adds such a folder to the renders.

Inputs should look like the images the head was trained and tested on: one part, cropped to a square
that it nearly fills, on a white background.

From Python:

    import sys; sys.path.insert(0, "recognizer")
    from predict import Recognizer
    rec = Recognizer()
    for r in rec.predict(["my_crop.jpg"]):
        print(r["label"], r["score"])
"""
import argparse, csv, glob, os, sys, warnings
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from torchvision import transforms

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_HEAD = os.path.join(REPO, "weights", "head_fgip18_seed0.pt")
DEFAULT_GALLERY = os.path.join(REPO, "weights", "gallery_fgip18_renders.npz")
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
# The FGIP data has no nut_f: its nut_g, nut_h, nut_i are the paper's nut f, nut g, nut h.
FGIP_PAPER_NAMES = {"nut_g": "nut_f", "nut_h": "nut_g", "nut_i": "nut_h"}

# Preprocessing and head are the ones in run_meta.py; keep the two files in step.
TFM = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class Head(nn.Module):
    def __init__(self, d=768, h=512, o=256):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, o))
        self.logit_scale = nn.Parameter(torch.tensor(10.0))
    def forward(self, x): return F.normalize(self.net(x), dim=-1)


def list_images(d):
    """Image files under d, sorted; searched recursively."""
    out = []
    for root, _, files in os.walk(d):
        out += [os.path.join(root, f) for f in files if f.lower().endswith(EXTS)]
    return sorted(out)


def class_folders(d):
    """{class name: [image paths]} for a folder laid out as d/<class>/<images>."""
    out = {}
    for c in sorted(os.listdir(d)):
        if os.path.isdir(os.path.join(d, c)):
            paths = list_images(os.path.join(d, c))
            if paths: out[c] = paths
    if not out: sys.exit(f"{d}: expected one sub-folder of images per class, found none")
    return out


def short(p):
    """A path relative to the current folder when it lies inside it, as typed otherwise."""
    try: r = os.path.relpath(p)
    except ValueError: return p              # another drive, on Windows
    return p if r.startswith("..") else r


def load_rgb(x):
    """A path or a PIL image -> RGB. Transparent pixels become white, the background the head was trained on."""
    im = x if isinstance(x, Image.Image) else Image.open(x)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA"); bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, im).convert("RGB")
    return im.convert("RGB")


class Recognizer:
    """Frozen DINOv2 ViT-B/14 + the released head + a reference set; predict() labels images by 1-NN."""

    def __init__(self, head=DEFAULT_HEAD, gallery=DEFAULT_GALLERY, refs=None, add_refs=None, device=None, batch=64):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu"); self.batch = batch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # the DINOv2 hub code warns that xFormers is absent; it is not needed
            self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True).to(self.dev).eval()
        self.head = Head().to(self.dev)
        self.head.load_state_dict(torch.load(head, map_location=self.dev, weights_only=True)); self.head.eval()
        self.head_path = head; self.display = {}; self.ref_desc = []
        feats, labels, classes, views = [], [], [], []
        if refs is None:                      # the shipped (or a saved) reference set, stored as DINOv2 features
            z = np.load(gallery, allow_pickle=False)
            classes = [str(c) for c in z["classes"]]
            feats.append(torch.from_numpy(z["feats"].astype(np.float32)))
            labels += [int(i) for i in z["labels"]]; views += [str(v) for v in z["views"]]
            if "display" in z.files: self.display = dict(zip(classes, (str(s) for s in z["display"])))
            self.ref_desc.append((gallery, len(z["labels"])))
        for d in ([refs] if refs else []) + ([add_refs] if add_refs else []):   # reference images in class folders
            n = 0
            for c, paths in class_folders(d).items():
                if c not in classes: classes.append(c)
                feats.append(self.features(paths)); labels += [classes.index(c)] * len(paths)
                views += [f"{c}/{os.path.basename(p)}" for p in paths]; n += len(paths)
            self.ref_desc.append((d, n))
        self.set_references(torch.cat(feats), labels, classes, views)

    def set_references(self, feats, labels, classes, views=None):
        """feats: (M, 768) L2-normalised DINOv2 features; labels: M class indices into classes."""
        self.classes = list(classes); self.ref_feats = feats.float()
        self.ref_labels = torch.as_tensor(labels, dtype=torch.long)
        self.ref_views = list(views) if views is not None else [""] * len(self.ref_labels)
        self.ref_emb = self.embed(self.ref_feats)
        self._cols = [(self.ref_labels == i).nonzero().flatten().to(self.dev) for i in range(len(self.classes))]

    @torch.no_grad()
    def features(self, images):
        """Paths or PIL images -> (N, 768) L2-normalised DINOv2 CLS features, on the CPU."""
        out = []
        for i in range(0, len(images), self.batch):
            ims = torch.stack([TFM(load_rgb(x)) for x in images[i:i + self.batch]]).to(self.dev)
            out.append(F.normalize(self.backbone.forward_features(ims)["x_norm_clstoken"], dim=-1).cpu())
        return torch.cat(out) if out else torch.zeros(0, 768)

    @torch.no_grad()
    def embed(self, feats):
        """(N, 768) DINOv2 features -> (N, 256) unit vectors in the head's space, on the device."""
        return torch.cat([self.head(feats[i:i + 4096].to(self.dev)) for i in range(0, len(feats), 4096)]) \
            if len(feats) else torch.zeros(0, 256, device=self.dev)

    @torch.no_grad()
    def predict(self, images, topk=3):
        """Label each image by its most similar reference view. Returns one dict per image:
        label, score (cosine similarity to the nearest view of that class), view (that view's name),
        topk [(class, score), ...] and, for the FGIP parts, paper_name."""
        res = []
        for i in range(0, len(images), self.batch):
            chunk = images[i:i + self.batch]
            sims = self.embed(self.features(chunk)) @ self.ref_emb.t()                       # (n, M)
            scores = torch.stack([sims[:, c].max(1).values for c in self._cols], dim=1)     # (n, classes)
            near = torch.stack([c[sims[:, c].argmax(1)] for c in self._cols], dim=1)        # nearest view per class
            top = scores.topk(min(topk, len(self.classes)), dim=1)
            for j, x in enumerate(chunk):
                k = int(scores[j].argmax())
                res.append({"image": x if isinstance(x, str) else "", "label": self.classes[k],
                            "paper_name": self.display.get(self.classes[k], ""),
                            "score": float(scores[j, k]), "view": self.ref_views[int(near[j, k])],
                            "topk": [(self.classes[int(c)], float(s)) for s, c in zip(top.values[j], top.indices[j])]})
        return res

    def save_gallery(self, path, fgip_names=False):
        """Write the current reference set as DINOv2 features, so it loads without the images next time."""
        d = dict(feats=self.ref_feats.numpy().astype(np.float16), labels=self.ref_labels.numpy().astype(np.int16),
                 classes=np.array(self.classes), views=np.array(self.ref_views), backbone=np.array("dinov2_vitb14"))
        if fgip_names: d["display"] = np.array([FGIP_PAPER_NAMES.get(c, c).replace("_", " ") for c in self.classes])
        np.savez_compressed(path, **d)


def main():
    pa = argparse.ArgumentParser(description="Label images with the released head by nearest reference view (1-NN).")
    pa.add_argument("images", nargs="*", help="image files, or folders of images")
    pa.add_argument("--eval", default="", help="folder laid out as DIR/<class>/<images>: report accuracy instead of labels")
    pa.add_argument("--head", default=DEFAULT_HEAD, help="head weights (default: weights/head_fgip18_seed0.pt)")
    pa.add_argument("--gallery", default=DEFAULT_GALLERY, help="reference set stored as DINOv2 features (default: the 18 FGIP parts' renders)")
    pa.add_argument("--refs", default="", help="use this folder of reference images, DIR/<class>/<images>, instead of --gallery")
    pa.add_argument("--add-refs", default="", help="add this folder of reference images to the reference set")
    pa.add_argument("--save-gallery", default="", help="write the reference set to this .npz so it can be reused with --gallery")
    pa.add_argument("--fgip-names", action="store_true", help="with --save-gallery: also store the paper's names of the FGIP parts")
    pa.add_argument("--topk", type=int, default=3, help="how many candidate classes to list per image")
    pa.add_argument("--csv", default="", help="also write the per-image results to this CSV file")
    pa.add_argument("--device", default="", help="cuda or cpu (default: cuda when available)")
    pa.add_argument("--batch", type=int, default=64)
    args = pa.parse_args()
    if not (args.images or args.eval or args.save_gallery): pa.error("give image files, --eval DIR, or --save-gallery FILE")
    for d in (args.refs, args.add_refs):
        if d and args.eval and os.path.realpath(d) == os.path.realpath(args.eval):
            sys.exit("the reference folder and the --eval folder are the same: every test image would be its own reference")

    rec = Recognizer(args.head, args.gallery, args.refs or None, args.add_refs or None, args.device or None, args.batch)
    print(f"head:       {short(args.head)}")
    srcs = " + ".join(f"{short(d)} ({n})" for d, n in rec.ref_desc) if len(rec.ref_desc) > 1 else short(rec.ref_desc[0][0])
    print(f"references: {srcs}: {len(rec.classes)} classes, {len(rec.ref_labels)} views")
    if args.save_gallery:
        rec.save_gallery(args.save_gallery, args.fgip_names); print(f"saved reference set -> {args.save_gallery}")

    truth = None
    if args.eval:
        folders = class_folders(args.eval)
        unknown = [c for c in folders if c not in rec.classes]
        if unknown: print(f"warning: no reference views for {', '.join(unknown)}; their images count as errors")
        paths = [p for c in folders for p in folders[c]]; truth = [c for c in folders for _ in folders[c]]
    else:
        paths = []
        for x in args.images:                 # a folder, a file, or a pattern the shell left unexpanded (Windows)
            paths += list_images(x) if os.path.isdir(x) else [x] if os.path.exists(x) else sorted(glob.glob(x)) or [x]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing: sys.exit("not found: " + ", ".join(missing[:5]))
    if not paths: return
    res = rec.predict(paths, args.topk)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow((["true_class"] if truth else []) + ["image", "prediction", "paper_name", "similarity", "nearest_view"]
                       + [f"{h}{i}" for i in range(1, args.topk + 1) for h in ("class_", "similarity_")])
            for n, r in enumerate(res):
                w.writerow(([truth[n]] if truth else []) + [r["image"], r["label"], r["paper_name"], f"{r['score']:.4f}", r["view"]]
                           + [v if isinstance(v, str) else f"{v:.4f}" for t in r["topk"] for v in t])
        print(f"wrote {args.csv}")

    if truth is None:
        wi = max(len(p) for p in paths); wc = max(len("prediction"), max(len(c) for c in rec.classes))
        named = any(r["paper_name"] for r in res)
        print(f"\n{'image':{wi}s}  {'prediction':{wc}s}  " + (f"{'paper name':10s}  " if named else "") + "cosine  next candidates")
        for r in res:
            rest = "  ".join(f"{c} {s:.3f}" for c, s in r["topk"][1:])
            print(f"{r['image']:{wi}s}  {r['label']:{wc}s}  " + (f"{r['paper_name']:10s}  " if named else "") + f"{r['score']:.3f}   {rest}")
        return

    pred = [r["label"] for r in res]; cls = sorted(set(truth))
    hit = {c: sum(p == t for p, t in zip(pred, truth) if t == c) for c in cls}; num = {c: truth.count(c) for c in cls}
    print(f"\n{len(truth)} images, {len(cls)} classes")
    print(f"accuracy {100 * sum(hit.values()) / len(truth):.2f}% ({sum(hit.values())}/{len(truth)});  "
          f"mean of the per-class accuracies {100 * np.mean([hit[c] / num[c] for c in cls]):.2f}%")
    wc = max(len(c) for c in cls + rec.classes)
    print(f"\n{'class':{wc}s}  images  correct   acc%  most frequent error")
    for c in cls:
        wrong = [p for p, t in zip(pred, truth) if t == c and p != c]
        err = max(set(wrong), key=wrong.count) if wrong else ""
        print(f"{c:{wc}s}  {num[c]:6d}  {hit[c]:7d}  {100 * hit[c] / num[c]:5.1f}  " + (f"{err} ({wrong.count(err)})" if err else ""))


if __name__ == "__main__":
    main()
