import numpy as np


def clamp_image(img):
    return np.minimum(np.maximum(img, 0.0), 1.0)


def center_crop_image(image, target_shapes):
    image = np.array(image)
    original_ndim = image.ndim
    if original_ndim == 2:
        image = image[..., None]
    height, width = image.shape[:2]
    target_h, target_w = target_shapes
    start_h = height // 2 - target_h // 2
    start_w = width // 2 - target_w // 2
    end_h = start_h + target_h
    end_w = start_w + target_w
    image = image[start_h:end_h, start_w:end_w]
    if original_ndim == 2:
        image = image[..., 0]
    return image


def overlay_image(image, color, ratio=0.5, mask=None):
    image = np.asarray(image, dtype=np.float32)
    color = np.asarray(color, dtype=np.float32)
    merged = image * (1 - ratio) + color * ratio
    if mask is None:
        return clamp_image(merged)
    output = image.copy()
    output[np.asarray(mask).astype(bool)] = merged[np.asarray(mask).astype(bool)]
    return clamp_image(output)
