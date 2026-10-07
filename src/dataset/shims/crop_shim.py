import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torch import Tensor

from ..types import AnyExample, AnyViews


def rescale(
    image: Float[Tensor, "c h_in w_in"],
    shape: tuple[int, int],
    scale: int,
    max_val: int,
    ori_dtype=np.uint8
) -> Float[Tensor, "c h_out w_out"]:
    h, w = shape
    if ori_dtype == np.uint8:
        image_new = (image * scale).clip(min=0, max=max_val)
        image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy().astype(ori_dtype)
        image_new = Image.fromarray(image_new)
        image_new = image_new.resize((w, h), Image.LANCZOS)
        image_new = np.array(image_new) / scale
        image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    elif ori_dtype == np.uint16:
        image_new = (image * scale).clip(min=0, max=max_val)
        image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy().astype(np.float32)
        resized_channels = []
        for channel in range(image_new.shape[2]):
            image_channel = Image.fromarray(image_new[..., channel], mode="F")
            image_channel = image_channel.resize((w, h), Image.LANCZOS)
            resized_channels.append(np.array(image_channel, dtype=np.float32))
        image_new = np.stack(resized_channels, axis=-1) / scale
        image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    else:
        raise NotImplementedError
    return rearrange(image_new, "h w c -> c h w")


def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2
    assert col >= 0, f"{w_in, w_out}"

    if row < 0:
        images = images[..., :, :, col : col + w_out] if h_in % 2 == 0 else images[..., :, 1:, col : col + w_out]
    else:
        # Center-crop the image.
        images = images[..., :, row : row + h_out, col : col + w_out]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy

    return images, intrinsics


def rescale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
    scale: int,
    max_val: int,
    ori_dtype=torch.uint8,
    align_width=False
) -> tuple[
    Float[Tensor, "*#batch c h_out w_out"],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    # When in width align model, width is fored to align, and height has a largest size
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    if not align_width:
        assert (h_out <= h_in and w_out <= w_in) or (h_out > h_in and w_out > w_in)

    # Support upscale
    # NOTE: This should only used for small upscale 230 - 256, always use larger data first
    scale_factor = max(h_out / h_in, w_out / w_in)
    # if scale_factor > 1.0:
    #     scale_factor = max(h_out / h_in, w_out / w_in)
    if align_width:
        scale_factor = w_out / w_in
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out

    # Reshape the images to the correct size. Assume we don't have to worry about
    # changing the intrinsics based on how the images are rounded.
    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = torch.stack([rescale(image, (h_scaled, w_scaled), scale, max_val, ori_dtype) for image in images])
    images = images.reshape(*batch, c, h_scaled, w_scaled)

    return center_crop(images, intrinsics, shape)


def apply_crop_shim_to_views(views: AnyViews, shape: tuple[int, int], align_width) -> AnyViews:
    images, intrinsics = rescale_and_crop(views["image"], views["intrinsics"], shape, 255, 255, np.uint8, align_width)
    normal, _ = rescale_and_crop(views["normal_gt"]*0.5+0.5, views["intrinsics"], shape, 255, 255, np.uint8, align_width)
    depth, _ = rescale_and_crop(views["depth_gt"][:, [0,0,0]], views["intrinsics"], shape, 1000, 65535, np.uint16, align_width)
    diffCol, _ = rescale_and_crop(views["diffCol_gt"], views["intrinsics"], shape, 255, 255, np.uint8, align_width)
    
    return {
        **views,
        "image": images,
        "normal_gt": normal * 2 - 1,
        "depth_gt": depth[:, 0],
        "diffCol_gt": diffCol,
        "intrinsics": intrinsics,
    }


def apply_crop_shim(example: AnyExample, shape: tuple[int, int], align_width: bool = False) -> AnyExample:
    """Crop images in the example."""
    return {
        **example,
        "context": apply_crop_shim_to_views(example["context"], shape, align_width),
        "target": apply_crop_shim_to_views(example["target"], shape, align_width),
    }
