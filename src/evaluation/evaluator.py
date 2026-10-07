"""Shared data, forward and reduction paths for released evaluation."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from torch import distributed as dist

from ..dataset import DATASETS, get_dataset
from ..dataset.view_sampler import get_view_sampler
from ..misc.validate_evaluation_result import arithmetic_mean
from ..misc.validate_evaluation_index import validate_index
from .camera_metrics import pose_metrics
from .metrics import compute_lpips, compute_psnr, compute_ssim


def load_yaml(path):
    cfg = OmegaConf.load(path)
    if "defaults" in cfg:
        del cfg["defaults"]
    return cfg


def build_dataset_cfg(args):
    dataset_cfg = load_yaml(Path("config/dataset") / f"{args.dataset}.yaml")
    if args.evaluation_config is not None:
        protocol = load_yaml(args.evaluation_config)
        dataset_cfg = OmegaConf.merge(dataset_cfg, protocol.dataset_overrides)
    sampler_cfg = load_yaml(Path("config/dataset/view_sampler") / f"{args.view_sampler}.yaml")

    dataset_cfg.view_sampler = sampler_cfg
    dataset_cfg.roots = [Path(root) for root in args.dataset_root]
    dataset_cfg.overfit_to_scene = args.scene
    dataset_cfg.cameras_are_circular = bool(
        dataset_cfg.get("cameras_are_circular", False)
    )
    dataset_cfg.supervision = "image"
    dataset_cfg.rotation_aug = False
    dataset_cfg.rotation_normal = bool(dataset_cfg.get("rotation_normal", False))
    dataset_cfg.first_frame_norm = bool(dataset_cfg.get("first_frame_norm", False))
    dataset_cfg.pose_scale_norm = bool(dataset_cfg.get("pose_scale_norm", False))
    dataset_cfg.pose_scale_euclidean = bool(dataset_cfg.get("pose_scale_euclidean", False))
    dataset_cfg.center_scale_head = bool(dataset_cfg.get("center_scale_head", False))

    if args.dataset_image_shape is not None:
        dataset_cfg.image_shape = list(args.dataset_image_shape)
    if args.dataset_shape is not None:
        dataset_cfg.dataset_shape = list(args.dataset_shape)
    if args.down_scale is not None:
        dataset_cfg.down_scale = args.down_scale
    if args.auto_rotate:
        dataset_cfg.auto_rotate = True
    if args.rotation_normal:
        dataset_cfg.rotation_normal = True
    if args.pose_scale_norm:
        dataset_cfg.pose_scale_norm = True
    if args.center_scale_head is not None:
        dataset_cfg.center_scale_head = args.center_scale_head == "true"
    if args.num_context_views is not None and "num_context_views" in sampler_cfg:
        dataset_cfg.view_sampler.num_context_views = args.num_context_views
    if args.num_target_views is not None and "num_target_views" in sampler_cfg:
        dataset_cfg.view_sampler.num_target_views = args.num_target_views
    if args.index_path is not None and "index_path" in sampler_cfg:
        dataset_cfg.view_sampler.index_path = str(args.index_path)
    if args.full_range_sample and "full_range_sample" in sampler_cfg:
        dataset_cfg.view_sampler.full_range_sample = True
    if args.randomize_context and "randomize_context" in sampler_cfg:
        dataset_cfg.view_sampler.randomize_context = True
    if args.colmap_dir is not None:
        dataset_cfg.colmap_dir = args.colmap_dir
    if args.image_dir is not None:
        dataset_cfg.image_dir = args.image_dir
    return dataset_cfg


def configure_evaluation(cfg):
    """Resolve a published preset without inheriting training preprocessing."""
    evaluation = cfg.evaluation
    preset = evaluation.preset
    views = evaluation.views
    matrix = evaluation.matrix[preset]
    if views not in matrix.views:
        raise ValueError(f"{preset} has no released {views}-view evaluation.")
    dataset = "dl3dv"
    args = SimpleNamespace(
        dataset=dataset,
        evaluation_config=None,
        view_sampler="evaluation",
        dataset_root=[evaluation.get("dataset_root") or matrix.dataset_root],
        scene=None,
        dataset_image_shape=matrix.image_shape,
        dataset_shape=[270, 480],
        down_scale=4,
        auto_rotate=False,
        rotation_normal=True,
        pose_scale_norm=True,
        center_scale_head=None,
        num_context_views=views,
        num_target_views=None,
        index_path=matrix.index_template.format(views=views),
        full_range_sample=True,
        randomize_context=False,
        colmap_dir=None,
        image_dir=None,
    )
    dataset_cfg = build_dataset_cfg(args)
    # Protocol normalization must precede explicit shape/sampler choices.
    dataset_cfg = OmegaConf.merge(dataset_cfg, evaluation.dataset_overrides)
    cfg.dataset = dataset_cfg
    cfg.seed = evaluation.runtime.seed
    cfg.model.encoder.lvspm.compile_model = evaluation.runtime.compile_model
    cfg.compile_model = False
    cfg.test.compute_scores = True
    cfg.data_loader.test.batch_size = 1
    cfg.data_loader.test.persistent_workers = False
    cfg.checkpointing.load = cfg.checkpointing.load or evaluation.checkpoints[matrix.checkpoint].path
    evaluation.task = matrix.task
    evaluation.index_path = args.index_path
    evaluation.aggregation = matrix.aggregation
    torch._dynamo.config.recompile_limit = evaluation.runtime.torch_compile_recompile_limit
    torch._dynamo.config.capture_scalar_outputs = False
    if evaluation.runtime.strict_protocol:
        index_path = Path(args.index_path)
        manifest_path = Path(matrix.get("index_manifest", evaluation.index_package.manifest))
        manifest = json.loads(manifest_path.read_text())
        record = next(item for item in manifest["files"] if item["path"] ==
                      str(index_path.relative_to(index_path.parent.parent)))
        validate_index(
            index_path, expected_contexts=views, expected_scenes=matrix.scenes,
            expected_sha256=record["sha256"], dataset_root=Path(args.dataset_root[0]),
            split="test",
        )


def make_evaluation_dataset(dataset_cfg, rank=0, world_size=1):
    """Partition scene keys once, without padding or a distributed sampler."""
    if world_size == 1:
        return get_dataset(dataset_cfg, "test", None, 0, 1)
    sampler = get_view_sampler(
        dataset_cfg.view_sampler, "test", False,
        dataset_cfg.cameras_are_circular, None,
    )
    keys = [scene for scene, entry in sampler.index.items() if entry is not None]
    sampler.index = {scene: sampler.index[scene] for scene in keys[rank::world_size]}
    return DATASETS[dataset_cfg.name](dataset_cfg, "test", sampler, 0, 1)


def score_output(output, batch, task):
    if task == "pose":
        return pose_metrics(output.extrinsic, batch["context"]["extrinsics"])
    rgb_gt, rgb = batch["target"]["image"][0], output.color[0]
    return {
        "psnr": compute_psnr(rgb_gt, rgb).mean().item(),
        "ssim": compute_ssim(rgb_gt, rgb).mean().item(),
        "lpips": compute_lpips(rgb_gt, rgb).mean().item(),
    }


def evaluate_batch(encoder, batch, task, compute_scores=True):
    with torch.no_grad():
        output = encoder(batch["context"], batch["target"])
        scores = score_output(output, batch, task) if compute_scores else None
    return output, scores


def finalize_evaluation(records, evaluation, output_dir):
    """Gather per-scene values, then reduce globally rather than averaging ranks."""
    if dist.is_initialized():
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, records)
        if dist.get_rank() != 0:
            return
    else:
        gathered = [records]
    merged = {}
    for rank_records in gathered:
        for scene, metrics in rank_records.items():
            if scene in merged:
                raise ValueError(f"Evaluation scene {scene} was produced more than once.")
            merged[scene] = metrics
    index = json.loads(Path(evaluation["index_path"]).read_text())
    expected = {scene for scene, entry in index.items() if entry is not None}
    if set(merged) != expected:
        raise ValueError("Evaluation scenes do not match the pinned index.")
    keys = set(next(iter(merged.values())))
    scene_mean = arithmetic_mean(list(merged.values()), keys)
    protocol_mean = scene_mean
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in (
        ("scores_all_avg.json", scene_mean),
        ("scores_protocol_avg.json", protocol_mean),
        ("evaluation_summary.json", {
            "task": evaluation["task"], "num_scenes": len(merged),
            "scenes": sorted(merged), "seed": evaluation["runtime"]["seed"],
            "compile_model": evaluation["runtime"]["compile_model"],
            "execution_path": "joint_context_target",
        }),
    ):
        (output_dir / filename).write_text(json.dumps(payload, indent=2, sort_keys=True))
