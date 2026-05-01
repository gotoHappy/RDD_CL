"""mydata change-detection dataset for training the cross-attention head.

Two sampling modes per object, mixed at the configured ratio:

    "anomaly pair" (probability = 1 − normal_pair_ratio)
        ref     = a random normal query   (from queries/)
        query   = a random anomaly query  (from queries/)
        gt_mask = the binary defect mask paired with that anomaly query

    "normal pair"  (probability = normal_pair_ratio)
        ref     = a random normal query   (from queries/)
        query   = a *different* random normal query
        gt_mask = all-zero mask (no defect anywhere)

The normal pairs are critical for de-biasing: without them, every training
target has at least some positive pixels, so the model can learn a "there
is always a defect somewhere" prior that shows up at test time as
spurious high-score regions on normal-vs-normal queries.

Augmentation:
    Spatial — same crop / flip applied synchronously to ref, query, gt
    Color   — independent brightness / contrast / JPEG quality on ref vs query
"""

import io
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as tvff

from .mydata_triplet import build_mydata_index


class MyDataChangeDetectDataset(Dataset):
    """Yields ``{ref, query, gt_mask, object_idx}`` per object.

    ``ref`` is a random normal query, ``query`` is a random anomaly query
    (with its real GT defect mask). All three share the same spatial crop /
    flip; ref and query use independent colour jitter.
    """

    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        mydata_root,
        figsize=(512, 512),
        spatial_scale=(0.85, 1.0),
        object_id_offset: int = 0,
        normal_pair_ratio: float = 0.3,
    ):
        self.mydata_root = Path(mydata_root)
        self.figsize = tuple(figsize)
        self.spatial_scale = tuple(spatial_scale)
        self.normal_pair_ratio = float(normal_pair_ratio)
        if not (0.0 <= self.normal_pair_ratio <= 1.0):
            raise ValueError(
                f"normal_pair_ratio must be in [0, 1], got {normal_pair_ratio}"
            )
        self.index = build_mydata_index(mydata_root)
        if not self.index:
            raise ValueError(f"No valid mydata objects under {mydata_root}")
        self.objects = sorted(self.index.keys())
        self.object_ids = {
            obj: object_id_offset + i for i, obj in enumerate(self.objects)
        }

    # ------------------------------------------------------------------

    def _load_rgb(self, path):
        return Image.open(path).convert("RGB")

    def _load_mask(self, path):
        if path is None or not Path(path).exists():
            return None
        arr = np.array(Image.open(path).convert("L"))
        binary = ((arr > 0).astype(np.uint8)) * 255
        return Image.fromarray(binary, mode="L")

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
            "brightness": random.uniform(0.85, 1.15),
            "contrast": random.uniform(0.85, 1.15),
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
        obj = self.objects[idx]
        entry = self.index[obj]

        # Decide pair type. Need ≥ 2 normal queries to make a *distinct*
        # normal pair; otherwise fall back to anomaly pair regardless of the
        # configured ratio.
        normals = entry["normal_queries"]
        is_normal_pair = (
            random.random() < self.normal_pair_ratio
            and len(normals) >= 2
        )

        if is_normal_pair:
            ref_path, qry_path = random.sample(normals, 2)
            mask_path = None  # all-zero gt mask
            is_anomalous = 0
        else:
            ref_path = random.choice(normals)
            qry_path, mask_path = random.choice(entry["anomaly_pairs"])
            is_anomalous = 1

        pil_ref = self._load_rgb(ref_path)
        pil_qry = self._load_rgb(qry_path)
        pil_gt = self._load_mask(mask_path) if mask_path is not None else None

        W, H = pil_ref.size
        if pil_gt is None:
            # Either it's a normal-normal pair, or the anomaly mask file was
            # missing — both treated as "no defect anywhere".
            pil_gt = Image.new("L", (W, H), color=0)

        # Spatial aug — synced across ref / query / gt
        top, left, ch, cw, flip = self._spatial_params(W, H)
        pil_ref = self._apply_spatial(pil_ref, top, left, ch, cw, flip)
        pil_qry = self._apply_spatial(pil_qry, top, left, ch, cw, flip)
        pil_gt = self._apply_spatial(pil_gt, top, left, ch, cw, flip, is_mask=True)

        gt_bin = (tvff.to_tensor(pil_gt).squeeze(0) > 0.5).float()

        cp_ref = self._sample_color_params()
        cp_qry = self._sample_color_params()
        pil_ref = self._apply_color(pil_ref, cp_ref)
        pil_qry = self._apply_color(pil_qry, cp_qry)

        return {
            "ref": self._to_tensor_norm(pil_ref),
            "query": self._to_tensor_norm(pil_qry),
            "gt_mask": gt_bin,
            "object_idx": torch.tensor(self.object_ids[obj], dtype=torch.long),
            "is_anomalous": torch.tensor(is_anomalous, dtype=torch.long),
        }

    def __len__(self):
        return len(self.objects)


def build_mydata_change_detect_loader(
    mydata_root,
    figsize=(512, 512),
    batch_size=8,
    num_workers=2,
    spatial_scale=(0.85, 1.0),
    object_id_offset: int = 0,
    normal_pair_ratio: float = 0.3,
):
    dataset = MyDataChangeDetectDataset(
        mydata_root=mydata_root,
        figsize=figsize,
        spatial_scale=spatial_scale,
        object_id_offset=object_id_offset,
        normal_pair_ratio=normal_pair_ratio,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
