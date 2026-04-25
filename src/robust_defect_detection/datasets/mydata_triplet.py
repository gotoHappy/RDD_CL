"""
mydata triplet dataset for contrastive training.

Uses REAL defect masks instead of online synthesis. Each __getitem__ returns
a triplet (N1, N2, A1) where:
    - N1 = ref.png (ideal-lighting normal reference)
    - N2 = a random normal query (different lighting, no anomaly)
    - A1 = a random anomaly query (real defect, with ground-truth defect_mask)

The foreground mask comes from ref_fg_mask.png (assumed valid for N1/N2/A1
since all three share the same object pose per mydata README §2–§3).

Output schema matches :class:`M2ADTripletDataset`:
    {
        "n1": (3, H, W),
        "n2": (3, H, W),
        "a1": (3, H, W),
        "fg_mask":     (H, W) float — binary foreground
        "defect_mask": (H, W) float — real GT defect (0/1)
        "object_idx":  scalar long
    }
"""

import io
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as tvff


_QUERY_RE = re.compile(
    r"^(?P<obj>obj_\d+)__"
    r"(?P<kind>normal|anomaly)__"
    r"(?P<defect>[^_]+(?:_[^_]+)*)__"
    r"light_(?P<light>\d+)\.png$"
)


def build_mydata_index(mydata_root):
    """
    Scan mydata/ and return:
        {
            obj_name: {
                "ref":            Path,
                "ref_fg_mask":    Path or None,
                "normal_queries": [Path, ...],      # normal images only
                "anomaly_pairs":  [(img_path, mask_path), ...],
            }
        }

    Only objects with at least one normal query AND one anomaly pair are kept.
    """
    mydata_root = Path(mydata_root)
    index = {}
    for obj_dir in sorted(mydata_root.iterdir()):
        if not obj_dir.is_dir() or not obj_dir.name.startswith("obj_"):
            continue
        ref = obj_dir / "ref.png"
        if not ref.exists():
            continue
        ref_fg = obj_dir / "ref_fg_mask.png"
        queries_dir = obj_dir / "queries"
        if not queries_dir.exists():
            continue

        normals = []
        anomalies = []
        for f in sorted(queries_dir.iterdir()):
            if f.suffix.lower() != ".png" or f.stem.endswith("_defect_mask"):
                continue
            m = _QUERY_RE.match(f.name)
            if m is None:
                continue
            if m.group("kind") == "normal":
                normals.append(f)
            else:  # anomaly
                mask = queries_dir / f"{f.stem}_defect_mask.png"
                if mask.exists():
                    anomalies.append((f, mask))

        if normals and anomalies:
            index[obj_dir.name] = {
                "ref": ref,
                "ref_fg_mask": ref_fg if ref_fg.exists() else None,
                "normal_queries": normals,
                "anomaly_pairs": anomalies,
            }
    return index


class MyDataTripletDataset(Dataset):
    """
    Samples triplets (ref, normal_query, anomaly_query) from mydata.

    Augmentation (same spirit as M2ADTripletDataset):
        Spatial (N1/N2/A1 share params; fg_mask & defect_mask co-transformed):
            - Random horizontal flip (p=0.5)
            - Random resize crop     (scale 0.8–1.0)
        Color  (N1 & A1 share params; N2 is independent):
            - Brightness ×[0.8, 1.2]
            - Contrast   ×[0.8, 1.2]
            - JPEG quality [70, 100]
    """

    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        mydata_root,
        figsize=(512, 512),
        object_id_offset=0,
        spatial_scale=(0.8, 1.0),
    ):
        self.mydata_root = Path(mydata_root)
        self.figsize = tuple(figsize)
        self.spatial_scale = tuple(spatial_scale)
        self.index = build_mydata_index(mydata_root)
        if not self.index:
            raise ValueError(f"No valid mydata objects found under {mydata_root}")
        self.units = sorted(self.index.keys())
        self.object_ids = {
            obj: object_id_offset + i for i, obj in enumerate(self.units)
        }
        self.n_objects = len(self.units)

    # ------------------------------------------------------------------

    def _load_rgb(self, path):
        return Image.open(path).convert("RGB")

    def _load_mask(self, path):
        if path is None or not Path(path).exists():
            return None
        arr = np.array(Image.open(path).convert("L"))
        binary = ((arr > 0).astype(np.uint8)) * 255
        return Image.fromarray(binary, mode="L")

    # ------------------------------------------------------------------

    def _spatial_params(self, W, H):
        lo, hi = self.spatial_scale
        scale = random.uniform(lo, hi)
        ch = int(H * scale)
        cw = int(W * scale)
        top = random.randint(0, H - ch)
        left = random.randint(0, W - cw)
        flip = random.random() < 0.5
        return top, left, ch, cw, flip

    def _apply_spatial(self, pil_img, top, left, ch, cw, flip, is_mask=False):
        fH, fW = self.figsize
        img = tvff.crop(pil_img, top, left, ch, cw)
        interp = Image.NEAREST if is_mask else Image.BILINEAR
        img = img.resize((fW, fH), interp)
        if flip:
            img = tvff.hflip(img)
        return img

    @staticmethod
    def _sample_color_params():
        return {
            "brightness": random.uniform(0.8, 1.2),
            "contrast": random.uniform(0.8, 1.2),
            "jpeg": random.randint(70, 100),
        }

    @staticmethod
    def _apply_color(pil_img, params):
        img = tvff.adjust_brightness(pil_img, params["brightness"])
        img = tvff.adjust_contrast(img, params["contrast"])
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=params["jpeg"])
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def _to_tensor_norm(self, pil_img):
        t = tvff.to_tensor(pil_img)
        return tvff.normalize(t, self._IMAGENET_MEAN, self._IMAGENET_STD)

    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        obj_name = self.units[idx]
        entry = self.index[obj_name]

        # Pool all normal images: ref + every normal query
        normal_pool = [entry["ref"]] + entry["normal_queries"]
        if len(normal_pool) >= 2:
            n1_path, n2_path = random.sample(normal_pool, 2)
        else:
            n1_path = n2_path = normal_pool[0]
        pil_n1 = self._load_rgb(n1_path)
        pil_n2 = self._load_rgb(n2_path)
        a1_path, mask_path = random.choice(entry["anomaly_pairs"])
        pil_a1 = self._load_rgb(a1_path)

        pil_fg = self._load_mask(entry["ref_fg_mask"])
        pil_def = self._load_mask(mask_path)

        W, H = pil_n1.size
        if pil_fg is None:
            pil_fg = Image.new("L", (W, H), color=255)
        if pil_def is None:
            pil_def = Image.new("L", (W, H), color=0)

        # --- Same spatial aug for N1 / N2 / A1 / fg_mask / defect_mask ---
        top, left, ch, cw, flip = self._spatial_params(W, H)
        pil_n1 = self._apply_spatial(pil_n1, top, left, ch, cw, flip)
        pil_n2 = self._apply_spatial(pil_n2, top, left, ch, cw, flip)
        pil_a1 = self._apply_spatial(pil_a1, top, left, ch, cw, flip)
        pil_fg = self._apply_spatial(pil_fg, top, left, ch, cw, flip, is_mask=True)
        pil_def = self._apply_spatial(pil_def, top, left, ch, cw, flip, is_mask=True)

        fg_bin = (tvff.to_tensor(pil_fg).squeeze(0) > 0.5).float()
        defect_mask = (tvff.to_tensor(pil_def).squeeze(0) > 0.5).float()

        # --- Color aug: N1 & A1 share params, N2 independent ---
        cp_n1 = self._sample_color_params()
        cp_n2 = self._sample_color_params()
        pil_n1 = self._apply_color(pil_n1, cp_n1)
        pil_a1 = self._apply_color(pil_a1, cp_n1)
        pil_n2 = self._apply_color(pil_n2, cp_n2)

        return {
            "n1": self._to_tensor_norm(pil_n1),
            "n2": self._to_tensor_norm(pil_n2),
            "a1": self._to_tensor_norm(pil_a1),
            "fg_mask": fg_bin,
            "defect_mask": defect_mask,
            "object_idx": torch.tensor(self.object_ids[obj_name], dtype=torch.long),
        }

    def __len__(self):
        return len(self.units)


def build_mydata_triplet_loader(
    mydata_root,
    figsize=(512, 512),
    batch_size=4,
    num_workers=2,
    object_id_offset=0,
    spatial_scale=(0.8, 1.0),
):
    dataset = MyDataTripletDataset(
        mydata_root=mydata_root,
        figsize=figsize,
        object_id_offset=object_id_offset,
        spatial_scale=spatial_scale,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
