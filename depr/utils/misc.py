# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from torch import Tensor
import numpy as np


def color_to_id(img):
    img = img.astype(np.uint32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return 256 * 256 * r + 256 * g + b


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


def nested_tensor_from_sequence_list(tensors: List[Tensor]):
    # List of tensors of shape (L, ...)
    dtype = tensors[0].dtype
    device = tensors[0].device
    max_size = max(t.shape[0] for t in tensors)
    batch_shape = [len(tensors), max_size]
    mask = torch.ones(batch_shape, dtype=torch.bool, device=device)
    batch_shape = batch_shape + list(tensors[0].shape)[1:]
    pad_tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    for b, t in enumerate(tensors):
        pad_tensor[b, : t.shape[0]] = t
        mask[b, : t.shape[0]] = False
    return NestedTensor(pad_tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(
            torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)
        ).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(
            img, (0, padding[2], 0, padding[1], 0, padding[0])
        )
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(
            m, (0, padding[2], 0, padding[1]), "constant", 1
        )
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def apply_transformations(mask: torch.Tensor, scale: tuple[float]) -> torch.Tensor:
    """
    Apply random transformations to a mask.
    """
    # hflip & scale
    if np.random.rand() > 0.5:
        mask = mask.flip(1)
    new_size = int(mask.shape[0] * np.random.uniform(*scale))
    transformed_mask = F.interpolate(
        mask[None, None, :, :].float(),
        size=(new_size, new_size),
        mode="bilinear",
        align_corners=False,
    )
    transformed_mask = transformed_mask.squeeze().round().bool()

    return transformed_mask


def place_mask_random(
    image: torch.Tensor, mask: torch.Tensor, scale_range: tuple[float]
) -> torch.Tensor:
    """
    Randomly place a mask onto an image with scale and horizontal flip augmentations.
    The operation is done in-place.

    Args:
        image: Tensor of shape (H, W, ...)
        mask: Tensor of shape (N, N)
        scale_range: Tuple of (min_scale, max_scale)

    Returns:
        Tuple of (masked_image, placed_mask) of shapes (H, W, C) and (H, W)
    """
    H, W = image.shape[:2]

    # Apply random transformations to mask
    mask = apply_transformations(mask, scale=scale_range)
    N = mask.shape[0]

    # Calculate valid ranges for top-left corner of mask
    # Allow mask to go partially outside the image
    x_min, x_max = -N // 2 + 1, W - N // 2 + 1
    y_min, y_max = -N // 2 + 1, H - N // 2 + 1

    # Randomly select top-left position
    x = np.random.randint(x_min, x_max - 1)
    y = np.random.randint(y_min, y_max - 1)

    # Calculate the overlapping region between mask and image
    x_start = max(0, x)
    y_start = max(0, y)
    x_end = min(x + N, W)
    y_end = min(y + N, H)

    # Calculate corresponding region in mask
    mask_x_start = max(0, -x)
    mask_y_start = max(0, -y)
    mask_x_end = mask_x_start + (x_end - x_start)
    mask_y_end = mask_y_start + (y_end - y_start)

    # Apply mask to the overlapping region
    mask_region = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
    image[y_start:y_end, x_start:x_end][mask_region] = 0

    return image


def random_occlude(
    sem_seg: torch.Tensor,
    mask: torch.Tensor,
    scale_range: tuple[float, float],
    occlusion_range: tuple[float, float],
    max_tries: int = 10,
) -> torch.Tensor:
    """
    Randomly occlude a semantic segmentation mask with a given mask.

    Args:
        sem_seg: Tensor of shape (H, W)
        mask: Tensor of shape (N, N)
        scale_range: Tuple of (min_scale, max_scale)
        occlusion_range: Tuple of (min_occlusion, max_occlusion)
        max_tries: Maximum number of tries before giving up

    Returns:
        Tensor of shape (H, W)
    """

    count_tries = 0

    while count_tries < max_tries:
        count_tries += 1
        masked_sem_seg = sem_seg.clone()
        masked_sem_seg = place_mask_random(masked_sem_seg, mask, scale_range)
        union = sem_seg.sum() - masked_sem_seg.sum()
        ratio = union / sem_seg.sum()
        if ratio >= occlusion_range[0] and ratio <= occlusion_range[1]:
            return masked_sem_seg

    return sem_seg
