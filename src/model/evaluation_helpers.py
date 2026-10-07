from pathlib import Path

from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..misc.image_io import save_image


def append_scores(
    output_store: dict[str, list],
    scores: dict,
    suffix: str = "",
    enabled: bool = True,
) -> None:
    if not enabled:
        return
    for key, value in scores.items():
        metric_name = f"{key}{suffix}"
        output_store.setdefault(metric_name, []).append(value)


def save_indexed_images(images, indices, path: Path) -> None:
    for index, color in zip(indices, images):
        save_image(color, path / f"{index:0>6}.png")


def trim_combined_target_views(batch, output) -> None:
    if batch["target"]["image"].shape[1] <= batch["target"]["extrinsics"].shape[1]:
        return
    n_target = batch["target"]["extrinsics"].shape[1]
    batch["target"]["image"] = batch["target"]["image"][:, -n_target:]
    batch["target"]["index"] = batch["target"]["index"][:, -n_target:]
    output.color = output.color[:, -n_target:]


def record_rgb_metrics(
    rgb_gt,
    rgb,
    batch_idx: int,
    eval_time_skip_steps: int,
    time_skip_steps: dict[str, int],
):
    if batch_idx < eval_time_skip_steps:
        time_skip_steps["encoder"] += 1

    return {
        "psnr": compute_psnr(rgb_gt, rgb).mean().item(),
        "ssim": compute_ssim(rgb_gt, rgb).mean().item(),
        "lpips": compute_lpips(rgb_gt, rgb).mean().item(),
    }
