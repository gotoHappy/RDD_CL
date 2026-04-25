import re
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvff


_LIGHT_PATTERN = re.compile(r"__light_(\d+)")


def _parse_light_id(path):
    match = _LIGHT_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"cannot parse light id from {path.name}")
    return int(match.group(1))


class ContrastiveTripletDataset(Dataset):
    def __init__(self, root, figsize=(504, 504)):
        self.root = Path(root)
        self.figsize = tuple(figsize)
        self.triplets = self._collect_triplets()
        if not self.triplets:
            raise ValueError(f"no valid contrastive triplets found under {root}")

    def _collect_triplets(self):
        triplets = []
        for obj_dir in sorted(self.root.glob("obj_*")):
            ref_path = obj_dir / "ref.png"
            fg_mask_path = obj_dir / "ref_fg_mask.png"
            query_dir = obj_dir / "queries"
            if not ref_path.exists() or not fg_mask_path.exists() or not query_dir.exists():
                continue

            normals = []
            anomalies_by_light = {}
            for query_path in sorted(query_dir.glob("*.png")):
                if query_path.name.endswith("_defect_mask.png"):
                    continue
                light_id = _parse_light_id(query_path)
                if "__normal__" in query_path.name:
                    normals.append({"path": query_path, "light_id": light_id})
                else:
                    mask_path = query_path.with_name(query_path.stem + "_defect_mask.png")
                    if not mask_path.exists():
                        continue
                    anomalies_by_light.setdefault(light_id, []).append(
                        {"path": query_path, "mask_path": mask_path, "light_id": light_id}
                    )

            for normal_1, normal_2 in permutations(normals, 2):
                for anomaly in anomalies_by_light.get(normal_1["light_id"], []):
                    triplets.append(
                        {
                            "object_id": obj_dir.name,
                            "ref_path": ref_path,
                            "fg_mask_path": fg_mask_path,
                            "normal_1_path": normal_1["path"],
                            "normal_2_path": normal_2["path"],
                            "anomaly_path": anomaly["path"],
                            "anomaly_mask_path": anomaly["mask_path"],
                            "light_id": normal_1["light_id"],
                        }
                    )
        return triplets

    def __len__(self):
        return len(self.triplets)

    def _load_rgb(self, path):
        image = Image.open(path).convert("RGB")
        image = tvff.resize(
            image,
            self.figsize,
            interpolation=InterpolationMode.BILINEAR,
        )
        return tvff.to_tensor(image)

    def _load_mask(self, path):
        image = Image.open(path)
        image = tvff.resize(
            image,
            self.figsize,
            interpolation=InterpolationMode.NEAREST,
        )
        mask = tvff.to_tensor(image).squeeze(0)
        return (mask > 0.0).float()

    def __getitem__(self, idx):
        sample = self.triplets[idx]
        return {
            "object_id": sample["object_id"],
            "light_id": sample["light_id"],
            "ref": self._load_rgb(sample["ref_path"]),
            "normal_1": self._load_rgb(sample["normal_1_path"]),
            "normal_2": self._load_rgb(sample["normal_2_path"]),
            "anomaly": self._load_rgb(sample["anomaly_path"]),
            "defect_mask": self._load_mask(sample["anomaly_mask_path"]),
            "fg_mask": self._load_mask(sample["fg_mask_path"]),
        }


def build_contrastive_triplet_loader(root, figsize, batch_size, num_workers, shuffle=True):
    dataset = ContrastiveTripletDataset(root=root, figsize=figsize)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )
