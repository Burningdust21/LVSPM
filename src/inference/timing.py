"""Synchronized inference timings, shared by folder inference and benchmarks."""

import statistics
import time

import torch


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(callable_, device: torch.device, repeats: int = 1):
    samples = []
    result = None
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        result = callable_()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return result, samples


def summarize(samples: list[float]) -> dict:
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def render_cached(encoder, state, extrinsics, intrinsics, chunk_size, device, warmup):
    """Warm each actual target shape, then time and return the requested frames."""
    targets = [
        {"extrinsics": extrinsics[:, start:start + chunk_size],
         "intrinsics": intrinsics[:, start:start + chunk_size]}
        for start in range(0, extrinsics.shape[1], chunk_size)
    ]
    warmed_shapes = set()
    for target in targets:
        count = target["extrinsics"].shape[1]
        if count not in warmed_shapes:
            for _ in range(warmup):
                encoder.render_targets(state, target)
            synchronize(device)
            warmed_shapes.add(count)

    images, samples, chunk_frames = [], [], []
    for target in targets:
        prediction, elapsed = timed(lambda: encoder.render_targets(state, target), device)
        images.append(prediction)
        samples.extend(elapsed)
        chunk_frames.append(target["extrinsics"].shape[1])
    total_ms = sum(samples)
    frames = sum(chunk_frames)
    return torch.cat(images, dim=1), {
        "frames": frames,
        "total_ms": total_ms,
        "ms_per_frame": total_ms / frames,
        "fps": frames * 1000.0 / total_ms,
        "chunk_frames": chunk_frames,
        "chunk_ms": samples,
    }
