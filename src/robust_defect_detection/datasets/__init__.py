import torch
import torch.utils.data
import torchvision.transforms as tvf
from torchvision.transforms import InterpolationMode

from .contrastive_triplet import build_contrastive_triplet_loader
from .mydata import MyDataDataset, MyTestDataDataset
from .. import torch_utils


_DATA_FACTORY = {
    "mydata": MyDataDataset,
    "mytestdata": MyTestDataDataset,
}


def get_dataset(name, root, **kwargs):
    if name not in _DATA_FACTORY:
        raise RuntimeError(f"unsupported dataset: {name}")
    return _DATA_FACTORY[name](root=root, **kwargs)


def prepare_transform(dataset, figsize=None):
    if figsize is None:
        figsize = (dataset.figsize // 14 * 14).tolist()
    transforms = [
        tvf.Resize(figsize, interpolation=InterpolationMode.BILINEAR),
        tvf.ToTensor(),
    ]
    transform = tvf.Compose(transforms)
    target_transform = tvf.Compose(
        [
            tvf.Resize(figsize, interpolation=InterpolationMode.NEAREST),
            tvf.ToTensor(),
            torch.squeeze,
        ]
    )
    return transform, target_transform


def build_dataloader(
    dataset,
    batch_size,
    num_workers,
    shuffle,
    return_ind=False,
    hflip_prob=0.0,
    figsize=None,
):
    transform, target_transform = prepare_transform(dataset, figsize=figsize)
    dataset = torch_utils.CDDataWrapper(
        dataset,
        transform=transform,
        target_transform=target_transform,
        return_ind=return_ind,
        hflip_prob=hflip_prob,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )


def get_eval_loader(root, batch_size=1, num_workers=1, figsize=None):
    dataset = get_dataset("mydata", root=root, mode="eval")
    return build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        return_ind=False,
        hflip_prob=0.0,
        figsize=figsize,
    )


def get_inference_loader(root, batch_size=1, num_workers=1, figsize=None):
    dataset = get_dataset("mytestdata", root=root)
    return build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        return_ind=False,
        hflip_prob=0.0,
        figsize=figsize,
    )
