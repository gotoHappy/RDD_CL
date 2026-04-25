import copy
from pathlib import Path

import numpy as np
from PIL import Image


class MyDataDataset:
    def __init__(self, root, mode="train"):
        mode = mode.lower()
        if mode not in {"train", "eval"}:
            raise ValueError(f"unsupported mode: {mode}")
        self.root = Path(root)
        self.mode = mode
        self.samples = self._collect_samples()
        if not self.samples:
            raise ValueError(f"no samples found under {self.root}")
        self._figsize = np.array(self.samples[0]["image_size"][::-1])

    def _collect_samples(self):
        samples = []
        for obj_dir in sorted(self.root.glob("obj_*")):
            ref_path = obj_dir / "ref.png"
            query_dir = obj_dir / "queries"
            if not ref_path.exists() or not query_dir.exists():
                continue
            for query_path in sorted(query_dir.glob("*.png")):
                if query_path.name.endswith("_defect_mask.png"):
                    continue
                mask_path = query_path.with_name(query_path.stem + "_defect_mask.png")
                is_normal = "__normal__" in query_path.name
                with Image.open(query_path) as img:
                    image_size = img.size
                samples.append(
                    {
                        "name": f"{obj_dir.name}/{query_path.name}",
                        "ref_path": ref_path,
                        "query_path": query_path,
                        "mask_path": mask_path if mask_path.exists() else None,
                        "is_normal": is_normal,
                        "image_size": image_size,
                    }
                )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ref = self._load_rgb(sample["ref_path"])
        query = self._load_rgb(sample["query_path"])
        mask = self._load_mask(sample["mask_path"], sample["image_size"])
        return ref, query, mask

    def loc(self, key):
        other = copy.copy(self)
        other.samples = list(np.array(self.samples, dtype=object)[key])
        return other

    def _load_rgb(self, path):
        image = Image.open(path).convert("RGB")
        image = np.asarray(image, dtype=np.float32) / 255.0
        return image

    def _load_mask(self, path, image_size):
        if path is None:
            width, height = image_size
            return np.zeros((height, width), dtype=np.float32)
        mask = Image.open(path)
        mask = np.asarray(mask, dtype=np.float32) / 255.0
        return (mask > 0.0).astype(np.float32)

    @property
    def filenames(self):
        return np.array([sample["name"] for sample in self.samples])

    @property
    def figsize(self):
        return self._figsize


class MyTestDataDataset:
    def __init__(self, root):
        self.root = Path(root)
        self.samples = self._collect_samples()
        if not self.samples:
            raise ValueError(f"no inference samples found under {self.root}")
        self._figsize = np.array(self.samples[0]["image_size"][::-1])

    def _collect_samples(self):
        samples = []
        for obj_dir in sorted(self.root.glob("obj_*")):
            ref_path = obj_dir / "ref.png"
            query_path = obj_dir / "query.png"
            if not ref_path.exists() or not query_path.exists():
                continue
            with Image.open(query_path) as img:
                image_size = img.size
            samples.append(
                {
                    "name": obj_dir.name,
                    "ref_path": ref_path,
                    "query_path": query_path,
                    "image_size": image_size,
                }
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ref = Image.open(sample["ref_path"]).convert("RGB")
        ref = np.asarray(ref, dtype=np.float32) / 255.0
        query = Image.open(sample["query_path"]).convert("RGB")
        query = np.asarray(query, dtype=np.float32) / 255.0
        width, height = sample["image_size"]
        mask = np.zeros((height, width), dtype=np.float32)
        return ref, query, mask

    @property
    def filenames(self):
        return np.array([sample["name"] for sample in self.samples])

    @property
    def figsize(self):
        return self._figsize
