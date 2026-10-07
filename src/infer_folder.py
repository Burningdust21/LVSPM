import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("JAXTYPING_DISABLE", "1")

import numpy as np
import torch
import torchvision.transforms as tf
from omegaconf import OmegaConf
from PIL import Image

from src.dataset.shims.crop_shim import rescale, center_crop
from src.inference.camera_export import visualize_camera_sets
from src.inference.interpolation import cubicspline_interpolate, interpolate_extrinsics, interpolate_intrinsics
from src.utils import load_encoder_weights, save_image_tensor, save_video, move_batch_to_device
from src.inference.timing import render_cached, summarize, synchronize, timed
from src.model.encoder.encoder_lvspm import EncoderLVSPM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LVSPM pose estimation and interpolated NVS on an image folder.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder-config", type=Path, default=Path("config/model/encoder/lvspm.yaml"))
    parser.add_argument("--image-shape", type=int, nargs=2, default=None, metavar=("H", "W"), help="Override the released checkpoint's input resolution.")
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=100.0)
    parser.add_argument("--num-input-views", type=int, default=None, help="Evenly sample this many images from the folder before model-input shuffling.")
    parser.add_argument("--model-input-seed", type=int, default=0, help="Fixed seed used to shuffle model input views.")
    parser.add_argument("--render-chunk-size", type=int, default=8, help="Number of target views rendered per forward pass after pose prediction.")
    parser.add_argument("--frames-per-segment", type=int, default=10)
    parser.add_argument("--num-anchor-views", type=int, default=16, help="Number of predicted views to use as interpolation anchors.")
    parser.add_argument("--save-cubic-video", action="store_true")
    parser.add_argument("--save-render-frames", action="store_true", help="Save rendered PNG frames in addition to videos.")
    parser.add_argument("--video-fps", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=1, help="Untimed warm-up passes for prefill and each actual render shape.")
    parser.add_argument("--disable-compile", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def resolve_image_shape(checkpoint: Path, image_shape=None) -> tuple[int, int]:
    if image_shape is not None:
        return tuple(image_shape)
    return (288, 512) if checkpoint.stem == "LVSPM_512x288" else (256, 448)


def list_images(image_dir: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in extensions)
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return paths


def natural_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return (int(digits), stem)
    return (10**12, stem)


def select_evenly_spaced_paths(image_paths: list[Path], num_input_views: int | None) -> list[Path]:
    if num_input_views is None or num_input_views >= len(image_paths):
        return image_paths
    if num_input_views < 2:
        raise ValueError("--num-input-views must be at least 2")
    sorted_paths = sorted(image_paths, key=natural_sort_key)
    indices = torch.linspace(0, len(sorted_paths) - 1, steps=num_input_views)
    indices = torch.round(indices).to(torch.long)
    indices[0] = 0
    indices[-1] = len(sorted_paths) - 1
    return [sorted_paths[idx] for idx in torch.unique_consecutive(indices).tolist()]


def load_rgb_image(path: Path) -> Image.Image | None:
    try:
        with Image.open(path) as image_file:
            image_file.load()
            image = image_file.convert("RGB")
    except Exception as exc:
        print(f"Skipping unreadable image: {path} ({exc})")
        return None
    if image.height > image.width:
        image = image.transpose(Image.Transpose.ROTATE_270)
    return image


def filter_valid_paths(image_paths: list[Path]) -> list[Path]:
    valid_paths = []
    for path in image_paths:
        if load_rgb_image(path) is not None:
            valid_paths.append(path)
    return valid_paths


def build_dummy_intrinsic(height: int, width: int) -> torch.Tensor:
    # Folder inference runs the unposed model and later uses predicted intrinsics.
    # This matrix only exists to satisfy the shared crop/preprocess path before prediction.
    intrinsic = torch.eye(3, dtype=torch.float32)
    intrinsic[0, 0] = 1.0
    intrinsic[1, 1] = 1.0
    intrinsic[0, 2] = 0.5
    intrinsic[1, 2] = 0.5
    return intrinsic


def normalize_intrinsics(intrinsics: torch.Tensor, height: int, width: int) -> torch.Tensor:
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, :] /= width
    intrinsics[..., 1, :] /= height
    return intrinsics


def load_and_preprocess_images(
    image_paths: list[Path],
    image_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    to_tensor = tf.ToTensor()
    images = []
    intrinsics = []
    for path in image_paths:
        image = load_rgb_image(path)
        if image is None:
            continue
        tensor = to_tensor(image)
        intrinsic = build_dummy_intrinsic(tensor.shape[1], tensor.shape[2])
        height, width = tensor.shape[-2:]
        factor = max(image_shape[0] / height, image_shape[1] / width)
        tensor = rescale(
            tensor, (round(height * factor), round(width * factor)),
            scale=255, max_val=255, ori_dtype=np.uint8,
        )
        tensor, intrinsic = center_crop(tensor, intrinsic, image_shape)
        images.append(tensor)
        intrinsics.append(intrinsic)

    if len(images) < 2:
        raise RuntimeError("Fewer than two valid images remain after skipping unreadable files")

    return torch.stack(images), torch.stack(intrinsics)


def build_seed_extrinsics(num_views: int) -> torch.Tensor:
    extrinsics = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(num_views, 1, 1)
    if num_views > 1:
        extrinsics[:, 0, 3] = torch.arange(num_views, dtype=torch.float32)
    return extrinsics


def build_batch(
    context_images: torch.Tensor,
    context_intrinsics: torch.Tensor,
    scene_name: str,
    near: float,
    far: float,
) -> dict[str, Any]:
    num_views = context_images.shape[0]
    # The initial pose pass only needs one dummy target view to satisfy the encoder interface.
    render_chunk_size = 1
    context_extrinsics = build_seed_extrinsics(num_views)
    dummy_target_images = context_images[:1].repeat(render_chunk_size, 1, 1, 1)
    dummy_target_intrinsics = context_intrinsics[:1].repeat(render_chunk_size, 1, 1)
    dummy_target_extrinsics = build_seed_extrinsics(render_chunk_size)
    context_indices = torch.arange(num_views, dtype=torch.int64)
    target_indices = torch.arange(render_chunk_size, dtype=torch.int64)
    return {
        "context": {
            "image": context_images.unsqueeze(0),
            "intrinsics": context_intrinsics.unsqueeze(0),
            "extrinsics": context_extrinsics.unsqueeze(0),
            "near": torch.full((1, num_views), near, dtype=torch.float32),
            "far": torch.full((1, num_views), far, dtype=torch.float32),
            "index": context_indices.unsqueeze(0),
        },
        "target": {
            "image": dummy_target_images.unsqueeze(0),
            "intrinsics": dummy_target_intrinsics.unsqueeze(0),
            "extrinsics": dummy_target_extrinsics.unsqueeze(0),
            "near": torch.full((1, render_chunk_size), near, dtype=torch.float32),
            "far": torch.full((1, render_chunk_size), far, dtype=torch.float32),
            "index": target_indices.unsqueeze(0),
        },
        "scene": [scene_name],
    }


def select_anchor_indices(num_views: int, num_anchor_views: int | None) -> torch.Tensor:
    if num_anchor_views is None or num_anchor_views >= num_views:
        return torch.arange(num_views, dtype=torch.long)
    if num_anchor_views < 2:
        raise ValueError("--num-anchor-views must be at least 2")
    indices = torch.linspace(0, num_views - 1, steps=num_anchor_views)
    indices = torch.round(indices).to(torch.long)
    indices[0] = 0
    indices[-1] = num_views - 1
    return torch.unique_consecutive(indices)


def make_model_input_order(num_views: int, seed: int) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randperm(num_views, generator=generator)


def build_interpolated_trajectory(
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    frames_per_segment: int,
    use_cubic_spline: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_extrinsics = []
    all_intrinsics = []
    t = torch.linspace(0, 1, frames_per_segment, dtype=torch.float32, device=extrinsics.device)
    for view_id in range(extrinsics.shape[0] - 1):
        all_extrinsics.append(interpolate_extrinsics(extrinsics[view_id], extrinsics[view_id + 1], t))
        all_intrinsics.append(interpolate_intrinsics(intrinsics[view_id], intrinsics[view_id + 1], t))
    trajectory_extrinsics = torch.cat(all_extrinsics, dim=0)
    trajectory_intrinsics = torch.cat(all_intrinsics, dim=0)
    if use_cubic_spline:
        trajectory_extrinsics = cubicspline_interpolate(extrinsics, trajectory_extrinsics)
    return trajectory_extrinsics, trajectory_intrinsics


def save_metadata(
    path: Path,
    image_paths: list[Path],
    predicted_extrinsics: torch.Tensor,
    interpolation_extrinsics: torch.Tensor,
    anchor_indices: torch.Tensor,
    sorted_render_indices: torch.Tensor,
) -> None:
    metadata = {
        "input_images": [path.name for path in image_paths],
        "num_input_images": len(image_paths),
        "sorted_render_order": sorted_render_indices.detach().cpu().tolist(),
        "anchor_indices": anchor_indices.detach().cpu().tolist(),
        "num_anchor_views": int(anchor_indices.numel()),
        "num_interpolated_frames": int(interpolation_extrinsics.shape[0]),
        "predicted_camera_centers": predicted_extrinsics[:, :3, 3].detach().cpu().tolist(),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    args = parse_args()
    args.image_shape = resolve_image_shape(args.checkpoint, args.image_shape)
    if args.frames_per_segment < 1:
        raise ValueError("--frames-per-segment must be >= 1")
    if args.warmup < 1 or args.render_chunk_size < 1:
        raise ValueError("--warmup and --render-chunk-size must be >= 1")

    print(f"Loading model from {args.checkpoint}")
    image_paths = list_images(args.image_dir)
    print(f"Reading images from {args.image_dir}")
    image_paths = filter_valid_paths(image_paths)
    image_paths = select_evenly_spaced_paths(image_paths, args.num_input_views)
    if len(image_paths) < 2:
        raise ValueError("At least two images are required for pose estimation and interpolation")
    num_views = len(image_paths)
    sorted_original_indices = torch.tensor(
        sorted(range(num_views), key=lambda idx: natural_sort_key(image_paths[idx])),
        dtype=torch.long,
    )
    model_input_order = make_model_input_order(num_views, args.model_input_seed)
    model_image_paths = [image_paths[idx] for idx in model_input_order.tolist()]
    inverse_model_order = torch.empty_like(model_input_order)
    inverse_model_order[model_input_order] = torch.arange(num_views, dtype=torch.long)
    sorted_render_indices = inverse_model_order[sorted_original_indices]

    device = torch.device(args.device)
    encoder_cfg = OmegaConf.load(args.encoder_config)
    if args.disable_compile:
        encoder_cfg.lvspm.compile_model = False
    encoder = EncoderLVSPM(encoder_cfg).to(device)
    load_encoder_weights(encoder, args.checkpoint)
    encoder.eval()
    print(f"Model loaded on {device}")

    context_images, context_intrinsics = load_and_preprocess_images(
        model_image_paths,
        tuple(args.image_shape),
    )
    print(f"Prepared {len(model_image_paths)} model input views")
    raw_batch = build_batch(
        context_images=context_images,
        context_intrinsics=context_intrinsics,
        scene_name=args.image_dir.name,
        near=args.near,
        far=args.far,
    )
    batch = move_batch_to_device(raw_batch, device)
    batch = encoder.get_data_shim()(batch)

    print(f"Running pose prediction with {batch['context']['image'].shape[1]} input views")
    with torch.no_grad():
        prefill = lambda: encoder.prefill_context(batch["context"])
        for _ in range(args.warmup):
            prefill()
        synchronize(device)
        prefill_state, prefill_samples = timed(prefill, device)
        predicted_extrinsics, predicted_intrinsics = encoder.decode_context_pose(prefill_state)
    if predicted_extrinsics is None or predicted_intrinsics is None:
        raise RuntimeError("Model did not return predicted poses for the image-folder input")

    predicted_extrinsics = predicted_extrinsics[0]
    predicted_extrinsics = torch.linalg.inv(predicted_extrinsics)
    predicted_intrinsics = predicted_intrinsics[0]
    image_height, image_width = batch["context"]["image"].shape[-2:]
    predicted_intrinsics_normalized = normalize_intrinsics(predicted_intrinsics, image_height, image_width)
    render_extrinsics = predicted_extrinsics[sorted_render_indices.to(predicted_extrinsics.device)]
    render_intrinsics_normalized = predicted_intrinsics_normalized[sorted_render_indices.to(predicted_intrinsics_normalized.device)]
    anchor_positions = select_anchor_indices(render_extrinsics.shape[0], args.num_anchor_views)
    anchor_indices = sorted_render_indices[anchor_positions]
    anchor_extrinsics = render_extrinsics[anchor_positions.to(render_extrinsics.device)]
    anchor_intrinsics_normalized = render_intrinsics_normalized[anchor_positions.to(render_intrinsics_normalized.device)]

    interpolated_extrinsics, interpolated_intrinsics = build_interpolated_trajectory(
        anchor_extrinsics,
        anchor_intrinsics_normalized,
        args.frames_per_segment,
    )
    print(
        f"Rendering interpolation video with {anchor_extrinsics.shape[0]} anchor views "
        f"and {interpolated_extrinsics.shape[0]} target frames"
    )
    with torch.no_grad():
        render_images, render_timing = render_cached(
            encoder, prefill_state,
            interpolated_extrinsics.unsqueeze(0),
            interpolated_intrinsics.unsqueeze(0),
            args.render_chunk_size, device, args.warmup,
        )

    output_dir = args.output_dir
    pose_dir = output_dir / "poses"
    pose_dir.mkdir(parents=True, exist_ok=True)
    if args.save_render_frames:
        render_dir = output_dir / "renders"
        render_dir.mkdir(parents=True, exist_ok=True)
        for frame_index, image in enumerate(render_images[0]):
            save_image_tensor(image, render_dir / f"{frame_index:06d}.png")
    save_video(render_images[0], output_dir / "render.mp4", args.video_fps)

    timings = {
        "measurement": "steady_state",
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "image_shape": [image_height, image_width],
        "input_views": num_views,
        "render_chunk_size": args.render_chunk_size,
        "compile_model": bool(encoder_cfg.lvspm.compile_model),
        "warmup": args.warmup,
        "excludes": ["compilation", "warmup", "image_loading", "pose_decoding", "video_encoding", "file_writes"],
        "prefill": summarize(prefill_samples),
        "render": render_timing,
    }
    if args.save_cubic_video:
        cubic_extrinsics, cubic_intrinsics = build_interpolated_trajectory(
            anchor_extrinsics,
            anchor_intrinsics_normalized,
            args.frames_per_segment,
            use_cubic_spline=True,
        )
        print(f"Rendering cubic interpolation video with {cubic_extrinsics.shape[0]} target frames")
        with torch.no_grad():
            cubic_images, timings["cubic_render"] = render_cached(
                encoder, prefill_state,
                cubic_extrinsics.unsqueeze(0),
                cubic_intrinsics.unsqueeze(0),
                args.render_chunk_size, device, args.warmup,
            )
        save_video(cubic_images[0], output_dir / "render_cubic.mp4", args.video_fps)

    np.save(pose_dir / "context_extrinsics.npy", predicted_extrinsics.detach().cpu().numpy())
    np.save(pose_dir / "context_intrinsics.npy", predicted_intrinsics.detach().cpu().numpy())

    camera_scene = visualize_camera_sets(
        torch.linalg.inv(render_extrinsics).detach().cpu().numpy(),
        None,
    )
    camera_scene.export(output_dir / "pose_visualization.glb")
    save_metadata(
        output_dir / "metadata.json",
        model_image_paths,
        predicted_extrinsics.detach().cpu(),
        interpolated_extrinsics.detach().cpu(),
        anchor_indices.detach().cpu(),
        sorted_render_indices.detach().cpu(),
    )
    with (output_dir / "timings.json").open("w", encoding="utf-8") as handle:
        json.dump(timings, handle, indent=2)
    timing_summary = (
        f"{image_width}x{image_height}, {num_views} input views, {timings['device_name']}: "
        f"warm prefill {timings['prefill']['mean_ms']:.2f} ms; "
        f"cached rendering {render_timing['fps']:.2f} FPS "
        f"({render_timing['frames']} frames, warm-up and I/O excluded)."
    )
    print(timing_summary)
    (output_dir / "timings.txt").write_text(timing_summary + "\n", encoding="utf-8")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
