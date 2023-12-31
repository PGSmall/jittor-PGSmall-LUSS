# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from PIL import Image
# import torch
# from torch.nn import functional as F
import jittor as jt
from jittor import nn
# from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from jittor.transform import resize, to_pil_image

from copy import deepcopy
from typing import Tuple


class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    # def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
    def apply_image_torch(self, image: jt.array) -> jt.array:

        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    # def apply_coords_torch(
    #     self, coords: torch.Tensor, original_size: Tuple[int, ...]
    # ) -> torch.Tensor:
    def apply_coords_torch(
        self, coords: jt.array, original_size: Tuple[int, ...]
    ) -> jt.array:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        # coords = deepcopy(coords).to(torch.float)
        coords = deepcopy(coords).to(jt.float32)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    # def apply_boxes_torch(
    #     self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    # ) -> torch.Tensor:
    def apply_boxes_torch(
        self, boxes: jt.array, original_size: Tuple[int, ...]
    ) -> jt.array:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
    

# def resize(image, size):
#     image = Image.fromarray(image)
#     image = image.resize(size, Image.BILINEAR)
#     image = jt.array(np.array(image))
#     return image

# def to_pil_image(image):
#     image_np = np.array(image).astype('uint8')
#     return Image.fromarray(image_np)

# def resize_jt(image, size):
#     # image = image.reshape(1, *image.shape)
#     return jt.nn.resize(image, size)

# def resize(image_pil, size):
#     image_pil = image_pil.resize(size, Image.BILINEAR) 
#     image_jt = jt.array(np.array(image_pil).astype('uint8'))

#     return image_jt


# def to_pil_image_jt(image):
#     image_np = image.astype('uint8')
#     return Image.fromarray(image_np)

