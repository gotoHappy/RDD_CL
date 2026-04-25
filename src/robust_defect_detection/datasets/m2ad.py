"""
M2AD dataset for contrastive training with online pseudo-anomaly synthesis.

Index structure built from meta_unsupervised.json:
    {cls_name: {obj_name: {view: [illumination_list]}}}

Each __getitem__ returns a triplet (N1, N2, A1) where:
    - N1, N2: same object, same view, different illuminations
    - A1: online synthetic anomaly applied to N1
"""

import io
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as tvff


# ---------------------------------------------------------------------------
# Octave noise (Perlin-like)
# ---------------------------------------------------------------------------

def _smooth_noise(h, w, scale, rng):
    gh = max(2, round(h / scale))
    gw = max(2, round(w / scale))
    grid = rng.random((gh, gw)).astype(np.float32)
    return np.array(Image.fromarray(grid).resize((w, h), Image.BICUBIC))


def perlin_like_noise(h, w, octaves=4, scale=4.0, seed=None):
    """Octave-stacked smooth noise, values in [0, 1]."""
    rng = np.random.default_rng(seed)
    noise = np.zeros((h, w), dtype=np.float32)
    amplitude, total = 1.0, 0.0
    cur_scale = float(scale)
    for _ in range(octaves):
        noise += amplitude * _smooth_noise(h, w, cur_scale, rng)
        total += amplitude
        amplitude *= 0.5
        cur_scale = max(cur_scale / 2.0, 1.0)
    return noise / total


# ---------------------------------------------------------------------------
# Pseudo-anomaly generator
# ---------------------------------------------------------------------------

class PseudoAnomalyGenerator:
    """
    Online pseudo-anomaly synthesis via five strategies:
        texture, cutpaste, color (HSV), noise, blur.

    Anomaly masks are generated with Perlin-like noise constrained to
    the foreground region.  Returns the anomaly image and the raw mask
    (at image resolution) for downstream patch-label computation.
    """

    def __init__(
        self,
        dtd_root=None,
        methods=None,
        perlin_octaves=(4, 6),
        perlin_scale=(2.0, 6.0),
        min_area_ratio=0.05,
        max_area_ratio=0.30,
        alpha_range=(0.4, 0.8),
        soft_sigma=3.0,
    ):
        self.dtd_root = Path(dtd_root) if dtd_root else None
        self.methods = methods or ["texture", "cutpaste", "color", "noise", "blur"]
        self.perlin_octaves = perlin_octaves
        self.perlin_scale = perlin_scale
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.alpha_range = alpha_range
        self.soft_sigma = soft_sigma
        self._dtd_paths = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_dtd_paths(self):
        if self._dtd_paths is not None:
            return
        if self.dtd_root is None:
            self._dtd_paths = []
            return
        img_dir = self.dtd_root / "images"
        paths = list(img_dir.glob("*/*.jpg")) + list(img_dir.glob("*/*.png"))
        self._dtd_paths = paths

    def _random_dtd(self, h, w):
        self._load_dtd_paths()
        if not self._dtd_paths:
            return None
        path = random.choice(self._dtd_paths)
        return np.array(Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)).astype(np.float32) / 255.0

    def _generate_mask(self, h, w, fg_np):
        """Perlin mask constrained to foreground, area ratio within [min, max]."""
        fg_area = max(float(fg_np.sum()), 1.0)
        for _ in range(20):
            oct = random.randint(*self.perlin_octaves)
            scale = random.uniform(*self.perlin_scale)
            noise = perlin_like_noise(h, w, octaves=oct, scale=scale)
            for thresh in np.linspace(0.3, 0.7, 9):
                m = (noise > thresh).astype(np.float32) * fg_np
                ratio = m.sum() / fg_area
                if self.min_area_ratio <= ratio <= self.max_area_ratio:
                    return m
        # Fallback: random rectangle within foreground bounding box
        rows, cols = np.where(fg_np > 0)
        if len(rows) == 0:
            rows, cols = np.arange(h), np.arange(w)
        r0, r1 = rows.min(), rows.max()
        c0, c1 = cols.min(), cols.max()
        rh = max(int((r1 - r0 + 1) * random.uniform(0.1, 0.3)), 1)
        rw = max(int((c1 - c0 + 1) * random.uniform(0.1, 0.3)), 1)
        ry = random.randint(r0, max(r0, r1 - rh))
        rx = random.randint(c0, max(c0, c1 - rw))
        m = np.zeros((h, w), dtype=np.float32)
        m[ry : ry + rh, rx : rx + rw] = 1.0
        return m * fg_np

    def _soft_mask(self, m_raw):
        ms = gaussian_filter(m_raw.astype(np.float64), sigma=self.soft_sigma)
        peak = ms.max()
        return (ms / peak).astype(np.float32) if peak > 0 else ms.astype(np.float32)

    # ------------------------------------------------------------------
    # Anomaly methods
    # ------------------------------------------------------------------

    def _texture(self, img, m_raw, m_soft):
        dtd = self._random_dtd(img.shape[0], img.shape[1])
        if dtd is None:
            return self._color(img, m_raw, m_soft)
        alpha = random.uniform(*self.alpha_range)
        w = (alpha * m_soft)[..., None]
        return np.clip(img * (1 - w) + dtd * w, 0.0, 1.0)

    def _cutpaste(self, img, m_raw, m_soft):
        H, W = img.shape[:2]
        rows, cols = np.where(m_raw > 0)
        if len(rows) == 0:
            return img
        r0, r1 = rows.min(), rows.max() + 1
        c0, c1 = cols.min(), cols.max() + 1
        rh, rw = max(r1 - r0, 1), max(c1 - c0, 1)
        # Source patch: random location sufficiently far from target
        for _ in range(10):
            sy = random.randint(0, max(H - rh, 0))
            sx = random.randint(0, max(W - rw, 0))
            if abs(sy - r0) + abs(sx - c0) > (rh + rw) // 4:
                break
        patch = img[sy : sy + rh, sx : sx + rw]
        ph, pw = patch.shape[:2]
        result = img.copy()
        th = min(r1, r0 + ph) - r0
        tw = min(c1, c0 + pw) - c0
        w = m_soft[r0 : r0 + th, c0 : c0 + tw][..., None]
        result[r0 : r0 + th, c0 : c0 + tw] = (
            img[r0 : r0 + th, c0 : c0 + tw] * (1 - w) + patch[:th, :tw] * w
        )
        return np.clip(result, 0.0, 1.0)

    @staticmethod
    def _rgb_to_hsv(img):
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        delta = maxc - minc + 1e-8
        s = np.where(maxc > 0, delta / maxc, 0.0)
        h = np.zeros_like(r)
        mr, mg, mb = maxc == r, maxc == g, maxc == b
        h[mr] = (((g - b)[mr] / delta[mr]) % 6.0) / 6.0
        h[mg] = ((b - r)[mg] / delta[mg] + 2.0) / 6.0
        h[mb] = ((r - g)[mb] / delta[mb] + 4.0) / 6.0
        return np.stack([h, s, maxc], axis=-1)

    @staticmethod
    def _hsv_to_rgb(hsv):
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        h6 = h * 6.0
        i = h6.astype(int) % 6
        f = h6 - np.floor(h6)
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, t, v])
        g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, v, v, q, p, p])
        b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, v, v, q])
        return np.stack([r, g, b], axis=-1).astype(np.float32)

    def _color(self, img, m_raw, m_soft):
        hsv = self._rgb_to_hsv(img)
        h_shift = random.uniform(-30.0 / 360, 30.0 / 360)
        s_scale = random.uniform(0.5, 1.5)
        v_scale = random.uniform(0.5, 1.5)
        p = hsv.copy()
        p[..., 0] = (p[..., 0] + h_shift) % 1.0
        p[..., 1] = np.clip(p[..., 1] * s_scale, 0.0, 1.0)
        p[..., 2] = np.clip(p[..., 2] * v_scale, 0.0, 1.0)
        perturbed = self._hsv_to_rgb(p)
        w = m_soft[..., None]
        return np.clip(img * (1 - w) + perturbed * w, 0.0, 1.0)

    def _noise(self, img, m_raw, m_soft):
        if random.random() < 0.5:
            sigma = random.uniform(30, 80) / 255.0
            noisy = np.clip(img + np.random.normal(0, sigma, img.shape).astype(np.float32), 0.0, 1.0)
        else:
            noisy = img.copy()
            sp = np.random.random(img.shape[:2]) < 0.08
            noisy[sp] = np.random.choice([0.0, 1.0], size=int(sp.sum()))[:, None]
        w = m_soft[..., None]
        return np.clip(img * (1 - w) + noisy * w, 0.0, 1.0)

    def _blur(self, img, m_raw, m_soft):
        radius = random.choice([5, 7, 10])
        blurred = np.array(
            Image.fromarray((img * 255).astype(np.uint8)).filter(
                __import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(radius=radius)
            )
        ).astype(np.float32) / 255.0
        w = m_soft[..., None]
        return np.clip(img * (1 - w) + blurred * w, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, img_tensor, fg_mask):
        """
        Args:
            img_tensor: (3, H, W) float32 [0, 1]
            fg_mask:    (H, W) binary float32 (1 = foreground)
        Returns:
            anomaly_tensor: (3, H, W) float32 [0, 1]
            defect_mask:    (H, W) float32 [0, 1]  (raw Perlin mask intensity)
        """
        H, W = img_tensor.shape[-2:]
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        fg_np = fg_mask.cpu().numpy()  # (H, W) float

        m_raw = self._generate_mask(H, W, fg_np)
        m_soft = self._soft_mask(m_raw)

        method = random.choice(self.methods)
        if method == "texture":
            result = self._texture(img_np, m_raw, m_soft)
        elif method == "cutpaste":
            result = self._cutpaste(img_np, m_raw, m_soft)
        elif method == "color":
            result = self._color(img_np, m_raw, m_soft)
        elif method == "noise":
            result = self._noise(img_np, m_raw, m_soft)
        else:
            result = self._blur(img_np, m_raw, m_soft)

        anomaly_tensor = torch.from_numpy(result).permute(2, 0, 1).float()
        defect_mask = torch.from_numpy(m_raw).float()
        return anomaly_tensor, defect_mask


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_m2ad_index(json_path, split="train"):
    """
    Returns nested dict:
        {cls_name: {obj_name: {view: [illumination_str, ...]}}}
    Only covers the requested split (default "train").
    """
    with open(json_path) as f:
        meta = json.load(f)

    index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for cls_name, entries in meta[split].items():
        for e in entries:
            index[cls_name][e["object_name"]][e["view"]].append(e["illumination"])
    return index


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class M2ADTripletDataset(Dataset):
    """
    Samples (N1, N2, A1) triplets from M2AD training split.

    N1, N2 — same object, same view, two different illuminations.
    A1     — online pseudo-anomaly applied to N1 (after spatial aug).

    Augmentation:
        Spatial (N1 & N2 share same random params; fg_mask is co-transformed):
            - Random horizontal flip  (p=0.5)
            - Random resize crop      (scale 0.8–1.0)
        Color  (N1 & A1 share same params; N2 is independent):
            - Brightness ×[0.8, 1.2]
            - Contrast   ×[0.8, 1.2]
            - JPEG quality [70, 100]
    """

    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        m2ad_root,
        json_path,
        mask_root=None,
        dtd_root=None,
        split="train",
        figsize=(512, 512),
        min_lights_per_view=2,
        # Perlin noise params
        perlin_octaves=(4, 6),
        perlin_scale=(2.0, 6.0),
        min_area_ratio=0.05,
        max_area_ratio=0.30,
    ):
        self.m2ad_root = Path(m2ad_root)
        self.mask_root = Path(mask_root) if mask_root else None
        self.figsize = tuple(figsize)

        # Build index
        self.index = build_m2ad_index(json_path, split)

        # Build sampling units & assign integer object IDs
        self.units = []       # [(cls_name, obj_name, view)]
        self.object_ids = {}  # (cls_name, obj_name) → int
        _oid = 0
        for cls_name in sorted(self.index):
            for obj_name in sorted(self.index[cls_name]):
                if (cls_name, obj_name) not in self.object_ids:
                    self.object_ids[(cls_name, obj_name)] = _oid
                    _oid += 1
                for view in sorted(self.index[cls_name][obj_name]):
                    if len(self.index[cls_name][obj_name][view]) >= min_lights_per_view:
                        self.units.append((cls_name, obj_name, view))

        if not self.units:
            raise ValueError(f"No valid M2AD triplet units found under {json_path} split={split}")

        self.n_objects = _oid
        self.anomaly_gen = PseudoAnomalyGenerator(
            dtd_root=dtd_root,
            perlin_octaves=perlin_octaves,
            perlin_scale=perlin_scale,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
        )

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _img_path(self, cls_name, obj_name, view, illumination):
        """Full path: m2ad_root / cls_name / cls_name/Good/obj_name/A{view}_I{illumination}.png"""
        rel = f"{cls_name}/Good/{obj_name}/A{view}_I{illumination}.png"
        return self.m2ad_root / cls_name / rel

    def _mask_path(self, cls_name, obj_name, view, illumination):
        """Full path: mask_root / cls_name/Good/obj_name/A{view}_I{illumination}.png"""
        rel = f"{cls_name}/Good/{obj_name}/A{view}_I{illumination}.png"
        return self.mask_root / rel

    def _load_image_pil(self, path):
        return Image.open(path).convert("RGB")

    def _load_fg_mask_pil(self, path):
        """Returns a PIL 'L' image with foreground as white, or None."""
        if self.mask_root is None:
            return None
        if not path.exists():
            return None
        mask = Image.open(path).convert("L")
        # BiRefNet masks: 0 = background, >0 = foreground
        arr = np.array(mask)
        binary = ((arr > 0).astype(np.uint8)) * 255
        return Image.fromarray(binary, mode="L")

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def _spatial_params(self, W, H):
        scale = random.uniform(0.8, 1.0)
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
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        cls_name, obj_name, view = self.units[idx]
        lights = self.index[cls_name][obj_name][view]
        light_a, light_b = random.sample(lights, 2)

        pil_n1 = self._load_image_pil(self._img_path(cls_name, obj_name, view, light_a))
        pil_n2 = self._load_image_pil(self._img_path(cls_name, obj_name, view, light_b))
        pil_fg = self._load_fg_mask_pil(self._mask_path(cls_name, obj_name, view, light_a))

        W, H = pil_n1.size
        if pil_fg is None:
            pil_fg = Image.new("L", (W, H), color=255)

        # --- Same spatial aug for N1, N2, fg_mask ---
        top, left, ch, cw, flip = self._spatial_params(W, H)
        pil_n1 = self._apply_spatial(pil_n1, top, left, ch, cw, flip, is_mask=False)
        pil_n2 = self._apply_spatial(pil_n2, top, left, ch, cw, flip, is_mask=False)
        pil_fg = self._apply_spatial(pil_fg, top, left, ch, cw, flip, is_mask=True)

        # Convert N1 to tensor for anomaly generation
        n1_raw = tvff.to_tensor(pil_n1)  # (3, H, W) float [0,1]
        fg_bin = (tvff.to_tensor(pil_fg).squeeze(0) > 0.5).float()  # (H, W)

        # --- Online pseudo-anomaly ---
        a1_raw, defect_mask = self.anomaly_gen(n1_raw, fg_bin)
        pil_a1 = tvff.to_pil_image(a1_raw)

        # --- Color aug: N1 & A1 share params, N2 is independent ---
        cp_n1 = self._sample_color_params()
        cp_n2 = self._sample_color_params()

        pil_n1 = self._apply_color(pil_n1, cp_n1)
        pil_a1 = self._apply_color(pil_a1, cp_n1)  # same as N1
        pil_n2 = self._apply_color(pil_n2, cp_n2)

        return {
            "n1": self._to_tensor_norm(pil_n1),           # (3, H, W)
            "n2": self._to_tensor_norm(pil_n2),           # (3, H, W)
            "a1": self._to_tensor_norm(pil_a1),           # (3, H, W)
            "fg_mask": fg_bin,                            # (H, W) binary float
            "defect_mask": defect_mask,                   # (H, W) Perlin intensity [0,1]
            "object_idx": torch.tensor(
                self.object_ids[(cls_name, obj_name)], dtype=torch.long
            ),
        }

    def __len__(self):
        return len(self.units)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_m2ad_triplet_loader(
    m2ad_root,
    json_path,
    mask_root=None,
    dtd_root=None,
    split="train",
    figsize=(518, 518),
    batch_size=8,
    num_workers=4,
    **dataset_kwargs,
):
    dataset = M2ADTripletDataset(
        m2ad_root=m2ad_root,
        json_path=json_path,
        mask_root=mask_root,
        dtd_root=dtd_root,
        split=split,
        figsize=figsize,
        **dataset_kwargs,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# Pair dataset — N1 & N2 only, no synthetic anomaly
# ---------------------------------------------------------------------------

class M2ADPairDataset(Dataset):
    """
    Samples (N1, N2) pairs from M2AD training split — **no** anomaly
    synthesis. Intended purely as a cross-illumination-invariance signal:
    same object, same view, two different illuminations.

    Output schema::

        {
            "n1": (3, H, W),
            "n2": (3, H, W),
            "fg_mask":    (H, W) float — binary foreground
            "object_idx": scalar long
        }

    Augmentation:
        Spatial (N1 / N2 / fg_mask share params):
            - Random horizontal flip  (p=0.5)
            - Random resize crop      (scale 0.8–1.0)
        Color  (independent params for N1 and N2 — each simulates its own
        lighting jitter on top of the real illumination difference):
            - Brightness ×[0.8, 1.2]
            - Contrast   ×[0.8, 1.2]
            - JPEG quality [70, 100]
    """

    _IMAGENET_MEAN = M2ADTripletDataset._IMAGENET_MEAN
    _IMAGENET_STD = M2ADTripletDataset._IMAGENET_STD

    def __init__(
        self,
        m2ad_root,
        json_path,
        mask_root=None,
        split="train",
        figsize=(512, 512),
        min_lights_per_view=2,
        object_id_offset=0,
    ):
        self.m2ad_root = Path(m2ad_root)
        self.mask_root = Path(mask_root) if mask_root else None
        self.figsize = tuple(figsize)

        self.index = build_m2ad_index(json_path, split)

        self.units = []
        self.object_ids = {}
        _oid = 0
        for cls_name in sorted(self.index):
            for obj_name in sorted(self.index[cls_name]):
                if (cls_name, obj_name) not in self.object_ids:
                    self.object_ids[(cls_name, obj_name)] = object_id_offset + _oid
                    _oid += 1
                for view in sorted(self.index[cls_name][obj_name]):
                    if len(self.index[cls_name][obj_name][view]) >= min_lights_per_view:
                        self.units.append((cls_name, obj_name, view))

        if not self.units:
            raise ValueError(f"No valid M2AD pair units found under {json_path} split={split}")
        self.n_objects = _oid

    # Reuse helpers from M2ADTripletDataset via composition.
    # Static methods must be re-wrapped with staticmethod() — direct attribute
    # aliasing pulls out the underlying function and re-binds `self`.
    _img_path = M2ADTripletDataset._img_path
    _mask_path = M2ADTripletDataset._mask_path
    _load_image_pil = M2ADTripletDataset._load_image_pil
    _load_fg_mask_pil = M2ADTripletDataset._load_fg_mask_pil
    _spatial_params = M2ADTripletDataset._spatial_params
    _apply_spatial = M2ADTripletDataset._apply_spatial
    _sample_color_params = staticmethod(M2ADTripletDataset._sample_color_params)
    _apply_color = staticmethod(M2ADTripletDataset._apply_color)
    _to_tensor_norm = M2ADTripletDataset._to_tensor_norm

    def __getitem__(self, idx):
        cls_name, obj_name, view = self.units[idx]
        lights = self.index[cls_name][obj_name][view]
        light_a, light_b = random.sample(lights, 2)

        pil_n1 = self._load_image_pil(self._img_path(cls_name, obj_name, view, light_a))
        pil_n2 = self._load_image_pil(self._img_path(cls_name, obj_name, view, light_b))
        pil_fg = self._load_fg_mask_pil(self._mask_path(cls_name, obj_name, view, light_a))

        W, H = pil_n1.size
        if pil_fg is None:
            pil_fg = Image.new("L", (W, H), color=255)

        # Spatial aug — synced across N1 / N2 / fg
        top, left, ch, cw, flip = self._spatial_params(W, H)
        pil_n1 = self._apply_spatial(pil_n1, top, left, ch, cw, flip, is_mask=False)
        pil_n2 = self._apply_spatial(pil_n2, top, left, ch, cw, flip, is_mask=False)
        pil_fg = self._apply_spatial(pil_fg, top, left, ch, cw, flip, is_mask=True)

        fg_bin = (tvff.to_tensor(pil_fg).squeeze(0) > 0.5).float()

        # Color aug — independent for each view, simulating extra illumination jitter
        cp_n1 = self._sample_color_params()
        cp_n2 = self._sample_color_params()
        pil_n1 = self._apply_color(pil_n1, cp_n1)
        pil_n2 = self._apply_color(pil_n2, cp_n2)

        return {
            "n1": self._to_tensor_norm(pil_n1),
            "n2": self._to_tensor_norm(pil_n2),
            "fg_mask": fg_bin,
            "object_idx": torch.tensor(
                self.object_ids[(cls_name, obj_name)], dtype=torch.long
            ),
        }

    def __len__(self):
        return len(self.units)


def build_m2ad_pair_loader(
    m2ad_root,
    json_path,
    mask_root=None,
    split="train",
    figsize=(512, 512),
    batch_size=4,
    num_workers=2,
    min_lights_per_view=2,
    object_id_offset=0,
):
    dataset = M2ADPairDataset(
        m2ad_root=m2ad_root,
        json_path=json_path,
        mask_root=mask_root,
        split=split,
        figsize=figsize,
        min_lights_per_view=min_lights_per_view,
        object_id_offset=object_id_offset,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
