import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def normalize_checkpoint_key(key):
    prefixes = (
        "encoder._orig_mod.", "encoder.", "_orig_mod.",
        "module.encoder.", "module.",
    )
    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def load_encoder_weights(encoder, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    normalized = {normalize_checkpoint_key(k): v for k, v in state_dict.items()}
    encoder.load_state_dict(normalized, strict=True)


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_device(v, device) for v in batch]
    if torch.is_tensor(batch):
        return batch.to(device)
    return batch


def save_image_tensor(image: torch.Tensor, path: Path) -> None:
    image = image.detach().cpu().clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().clamp(0, 1)
    return (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def save_video(images: torch.Tensor, path: Path, fps: int) -> None:
    frames = [tensor_to_uint8(image) for image in images]
    if not frames:
        return
    height, width = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for frame in frames:
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr is not None else ""
        return_code = process.wait()
    finally:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg video export failed for {path}:\n{stderr}")
