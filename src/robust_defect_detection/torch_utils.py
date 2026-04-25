import numpy as np
import torch
import torchvision.transforms.functional as tvff
from torchvision import transforms as tvf


def translate_image(tx, ty):
    def f(img):
        original_shape = img.shape
        if len(original_shape) == 2:
            img = img.unsqueeze(0)
        out = tvf.functional.affine(img, angle=0, translate=(tx, ty), scale=1.0, shear=0)
        if len(original_shape) == 2:
            out = out.squeeze(0)
        return out

    return f


def rotate_image(angle):
    def f(img):
        original_shape = img.shape
        if len(original_shape) == 2:
            img = img.unsqueeze(0)
        out = tvf.functional.affine(img, angle=angle, translate=(0, 0), scale=1.0, shear=0)
        if len(original_shape) == 2:
            out = out.squeeze(0)
        return out

    return f


class CDDataWrapper:
    def __init__(
        self,
        dataset,
        transform=None,
        target_transform=None,
        return_ind=False,
        translate0=(0, 0),
        translate1=(0, 0),
        rotate_angle0=0.0,
        rotate_angle1=0.0,
        hflip_prob=0.0,
        augment_diff_degree=None,
        augment_diff_translate=None,
    ):
        self.dataset = dataset
        self.transform = transform if transform is not None else (lambda x: x)
        self.target_transform = target_transform if target_transform is not None else (lambda x: x)
        self.return_ind = return_ind
        self.hflip_prob = hflip_prob
        self.translate0 = translate_image(*translate0)
        self.translate1 = translate_image(*translate1)
        self.rotate0 = rotate_image(rotate_angle0)
        self.rotate1 = rotate_image(rotate_angle1)
        self._pre_transform = tvf.ToTensor()
        self._pos_transform = tvf.ToPILImage()
        self.augment_diff_degree = 0.0 if augment_diff_degree is None else abs(augment_diff_degree)
        self.augment_diff_translate = augment_diff_translate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        t0, t1, gt = self.dataset[idx]
        t0 = self._pre_transform(t0)
        t1 = self._pre_transform(t1)
        gt = self._pre_transform(gt)
        t0 = self.translate0(t0)
        t1 = self.translate1(t1)
        gt = self.translate0(gt)
        t0 = self.rotate0(t0)
        t1 = self.rotate1(t1)
        gt = self.rotate0(gt)
        t0 = self._pos_transform(t0)
        t1 = self._pos_transform(t1)
        gt = self._pos_transform(gt)
        t0 = self.transform(t0)
        t1 = self.transform(t1)
        gt = self.target_transform(gt)

        if self.augment_diff_degree > 0.0:
            degree = np.random.uniform(-self.augment_diff_degree, self.augment_diff_degree)
            t0 = rotate_image(degree)(t0)
            gt = rotate_image(degree)(gt)

        if self.augment_diff_translate is not None:
            translate = np.random.uniform(
                self.augment_diff_translate[0],
                self.augment_diff_translate[1],
                size=2,
            )
            t0 = translate_image(*translate)(t0)
            gt = translate_image(*translate)(gt)

        if np.random.random() < self.hflip_prob:
            t0 = tvff.hflip(t0)
            t1 = tvff.hflip(t1)
            gt = tvff.hflip(gt)

        output = (t0, t1, gt)
        if self.return_ind:
            return idx, output
        return output
