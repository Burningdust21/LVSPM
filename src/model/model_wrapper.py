
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import numpy as np
import torch
import wandb
from einops import pack, rearrange
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.camera_metrics import pose_metrics
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..evaluation.evaluator import evaluate_batch, finalize_evaluation
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image
from ..misc.LocalLogger import LocalLogger
from ..misc.step_tracker import StepTracker
from ..misc.tensor_memory import retained_tensor_memory
from ..misc.visual_util import visualize_camera
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    cubicspline_interpolate,
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble_transformation,
)
from ..visualization.layout import add_border, hcat, vcat

from .encoder import Encoder
from .types import LVSPM
from .evaluation_helpers import (
    append_scores,
    record_rgb_metrics,
    save_indexed_images,
    trim_combined_target_views,
)

module_logger = logging.getLogger(__name__)


@dataclass
class OptimizerCfg:
    lr: float
    grad_clip: float
    grad_clip_steps: int
    weight_decay: float
    scale_lr: bool

@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    eval_time_skip_steps: int
    save_artifacts: bool = False
    save_image: bool = False
    save_video: bool = False
    video_trajectory: str = "both"
    max_video_anchor: int = 32
    extrpolation_src: str = "target"
    split_prefill_render: bool = False
    split_warmup: int = 1
    split_repetitions: int = 3
    checkpoint_sha256: str = ""


@dataclass
class TrainCfg:
    extended_visualization: bool
    print_log_every_n_steps: int


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        ...


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None
    compile_model: bool

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        compile_model: bool,
        evaluation_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker
        self.compile_model = compile_model
        self.evaluation_cfg = evaluation_cfg
        self.evaluation_records = {}
        if evaluation_cfg is not None:
            self.allow_zero_length_dataloader_with_multiple_devices = True

        # Set up the model.
        self.encoder = encoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.test_scenes: list[str] = []

        self.time_skip_steps_dict = {"encoder": 0}
        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
        if self.test_cfg.split_prefill_render:
            if self.test_cfg.compute_scores:
                raise ValueError("Split timing must use test.compute_scores=false")
            if self.test_cfg.split_warmup < 0 or self.test_cfg.split_repetitions < 1:
                raise ValueError("Split timing requires warmup >= 0 and repetitions >= 1")
            self.time_skip_steps_dict.update(
                {"prefill": 0, "render": 0, "prefill_render_total": 0}
            )
            self._split_metadata = None


    def configure_model(self):
        # Compile after DDP
        # dynamic=True will triger some leaking in typing function
        if self.compile_model and not self.trainer.testing:
            self.encoder = torch.compile(self.encoder)

    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        output = self.encoder(
            batch["context"],
            batch["target"],
        )
        target_gt = batch["target"]["image"]

        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean(), prog_bar=True)

        # Compute and log loss.
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(output, batch, self.global_step)
            self.log(f"loss/{loss_fn.name}", loss, prog_bar=True)
            total_loss = total_loss + loss
        self.log("loss/total", total_loss, prog_bar=True)
        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            module_logger.info(
                "train step %s; rank %s; scene=%s; context=%s; bound=[%s %s]",
                self.global_step,
                self.global_rank,
                [x[:20] for x in batch["scene"]],
                batch["context"]["index"].tolist(),
                batch["context"]["near"].detach().cpu().numpy().mean(),
                batch["context"]["far"].detach().cpu().numpy().mean(),
            )

        self.log("info/near", batch["context"]["near"].detach().cpu().numpy().mean())
        self.log("info/far", batch["context"]["far"].detach().cpu().numpy().mean())
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss

    def _append_test_scores(self, scores: dict, suffix: str = "") -> None:
        append_scores(
            self.test_step_outputs,
            scores,
            suffix,
            enabled=self.test_cfg.compute_scores,
        )

    def _save_indexed_images(self, images, indices, path: Path) -> None:
        save_indexed_images(images, indices, path)

    def _trim_combined_target_views(self, batch: BatchedExample, output) -> None:
        trim_combined_target_views(batch, output)

    def _run_test_forward(self, batch: BatchedExample):
        if self.test_cfg.split_prefill_render:
            return self._run_split_test_forward(batch)
        with self.benchmarker.time("encoder"):
            output = self.encoder(
                batch["context"],
                batch["target"],
            )
        self._trim_combined_target_views(batch, output)
        return output

    def _run_split_test_forward(self, batch: BatchedExample):
        for _ in range(self.test_cfg.split_warmup):
            state = self.encoder.prefill_context(batch["context"])
            self.encoder.render_targets(state, batch["target"])

        output = None
        for _ in range(self.test_cfg.split_repetitions):
            with self.benchmarker.time("prefill_render_total"):
                with self.benchmarker.time("prefill"):
                    state = self.encoder.prefill_context(batch["context"])
                with self.benchmarker.time("render"):
                    color = self.encoder.render_targets(state, batch["target"])
                    extrinsic, intrinsic = self.encoder.decode_context_pose(state)
                    output = LVSPM(color, extrinsic, intrinsic, state.pose_encoding)
        if output is None:
            raise RuntimeError("Split rendering produced no output")
        state_memory = retained_tensor_memory(state)
        image = batch["context"]["image"]
        self._split_metadata = {
            "context_views": int(image.shape[1]),
            "target_views": int(batch["target"]["image"].shape[1]),
            "resolution": [int(image.shape[-1]), int(image.shape[-2])],
            "checkpoint_sha256": self.test_cfg.checkpoint_sha256,
            "gpu": torch.cuda.get_device_name(image.device) if image.is_cuda else "cpu",
            "dtype": str(image.dtype).removeprefix("torch."),
            "attention_dtype": str(
                self.encoder.view_predictor.attention_dtype
            ).removeprefix("torch."),
            "warmup": self.test_cfg.split_warmup,
            "repetitions": self.test_cfg.split_repetitions,
            "prefill_state_resident_bytes": state_memory.resident_bytes,
            "prefill_state_unique_storages": state_memory.unique_storages,
            "prefill_state_tensor_references": state_memory.tensor_references,
            "prefill_state_memory_semantics": (
                "sum of unique underlying tensor storage bytes retained by the "
                "reusable prefill state; aliases/views are counted once"
            ),
        }
        return output

    def _print_test_batch(self, batch: BatchedExample) -> None:
        module_logger.info(
            "Testing step %s; scene=%s; context=%s; bound=[%s %s]",
            self.global_step,
            [x[:20] for x in batch["scene"]],
            batch["context"]["index"].tolist(),
            batch["context"]["near"].detach().cpu().numpy().mean(),
            batch["context"]["far"].detach().cpu().numpy().mean(),
        )
        module_logger.info("On test batch: %s", batch["scene"])

    def _save_camera_reference_files(self, batch: BatchedExample, predicted_extrinsic, scene_path: Path) -> None:
        pose_dict = {}
        for idx in range(batch["context"]["extrinsics"][0].shape[0]):
            pose_idx = batch["context"]["index"][0][idx]
            pose_dict[f"{pose_idx:06d}.png"] = batch["context"]["extrinsics"][0][idx].cpu().numpy()
        for idx in range(batch["target"]["extrinsics"][0].shape[0]):
            pose_idx = batch["target"]["index"][0][idx]
            pose_dict[f"{pose_idx:06d}.png"] = batch["target"]["extrinsics"][0][idx].cpu().numpy()
        np.save(scene_path / "cameras.npy", pose_dict)

        pose_dict = {}
        for idx in range(batch["context"]["extrinsics"][0].shape[0]):
            pose_idx = batch["context"]["index"][0][idx]
            pose_dict[f"{pose_idx:06d}.png"] = np.linalg.inv(predicted_extrinsic[0][idx].cpu().numpy())
        np.save(scene_path / "cameras_pred_inputs.npy", pose_dict)

    def _save_pose_outputs(
        self,
        batch: BatchedExample,
        predicted_extrinsic,
        scene: str,
        scene_path: Path,
        metric_suffix: str,
        file_suffix: str,
    ) -> None:
        metrics = pose_metrics(predicted_extrinsic, batch["context"]["extrinsics"])
        self._append_test_scores(metrics, metric_suffix)
        with (scene_path / f"metrics{file_suffix}_pose.json").open("w") as f:
            json.dump(metrics, f)

        if not self.test_cfg.save_artifacts:
            return

        glbscene = visualize_camera(predicted_extrinsic[0], batch["context"]["extrinsics"][0])
        glbscene.export(file_obj=scene_path / f"cameras{file_suffix}_{scene}.ply")
        glbscene = visualize_camera(predicted_extrinsic[0], batch["context"]["extrinsics"][0], align=True)
        glbscene.export(file_obj=scene_path / f"cameras{file_suffix}_align_{scene}.ply")

        if file_suffix == "":
            self._save_camera_reference_files(batch, predicted_extrinsic, scene_path)

    def _save_rgb_outputs(self, batch: BatchedExample, images_prob, scene_path: Path) -> None:
        if not self.test_cfg.save_image:
            return
        self._save_indexed_images(
            images_prob,
            batch["target"]["index"][0],
            scene_path / "color",
        )
        self._save_indexed_images(
            batch["context"]["image"][0],
            batch["context"]["index"][0],
            scene_path / "color_input",
        )
        self._save_indexed_images(
            batch["target"]["image"][0],
            batch["target"]["index"][0],
            scene_path / "color_gt",
        )

    def _render_eval_videos(self, batch: BatchedExample, scene: str, scene_path: Path) -> None:
        module_logger.info("Saving eval video for %s", scene)
        if batch["context"]["image"].shape[1] == 1:
            batch["context"] = batch["target"]
            batch["context"]["image"] = batch["context"]["image"].expand(
                -1,
                batch["target"]["extrinsics"].shape[1],
                -1,
                -1,
                -1,
            )

        source = batch[self.test_cfg.extrpolation_src]
        if source["image"].shape[1] > self.test_cfg.max_video_anchor:
            sample_interval = math.floor(source["image"].shape[1] / self.test_cfg.max_video_anchor)
        else:
            sample_interval = 1

        if not self.test_cfg.save_video:
            return

        trajectory = self.test_cfg.video_trajectory
        if trajectory not in {"linear", "spline", "both"}:
            raise ValueError(
                "test.video_trajectory must be one of: linear, spline, both"
            )

        try:
            if trajectory in {"linear", "both"}:
                self.render_video_interpolation_all(
                    batch,
                    scene_path / "video_linear",
                    sample_interval,
                    use_cubic_spline=False,
                )
            if trajectory in {"spline", "both"}:
                self.render_video_interpolation_all(
                    batch,
                    scene_path / "video_spline",
                    sample_interval,
                    use_cubic_spline=True,
                )
        except Exception as exc:
            module_logger.warning("Skipping eval video for %s: %r", scene, exc)

    def _record_rgb_metrics(self, rgb_gt, rgb, batch_idx: int) -> dict:
        if not self.test_cfg.compute_scores:
            return {}
        metrics = record_rgb_metrics(
            rgb_gt,
            rgb,
            batch_idx,
            self.test_cfg.eval_time_skip_steps,
            self.time_skip_steps_dict,
        )
        self._append_test_scores(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        if self.evaluation_cfg is not None:
            self._test_evaluation_batch(batch)
            return
        b = batch["target"]["image"].shape[0]
        assert b == 1

        output = self._run_test_forward(batch)
        if self.test_cfg.split_prefill_render:
            return
        self._print_test_batch(batch)
        (scene,) = batch["scene"]
        if scene in self.test_scenes:
            raise RuntimeError(f"Evaluation scene {scene} was produced more than once.")
        self.test_scenes.append(scene)
        path = self.test_cfg.output_path / get_cfg()["wandb"]["name"]
        scene_path = path / scene
        scene_path.mkdir(exist_ok=True, parents=True)

        if hasattr(output, "extrinsic") and output.extrinsic is not None:
            self._save_pose_outputs(batch, output.extrinsic, scene, scene_path, "", "")

        if not hasattr(output, "color"):
            module_logger.info("Testing complete: %s", scene)
            return

        images_prob = output.color[0]
        rgb_gt = batch["target"]["image"][0]
        module_logger.info("Saving images for %s at %s", scene, path)
        self._save_rgb_outputs(batch, images_prob, scene_path)
        self._render_eval_videos(batch, scene, scene_path)

        metrics = self._record_rgb_metrics(rgb_gt, images_prob, batch_idx)
        if metrics:
            with (scene_path / "metrics.json").open("w") as f:
                json.dump(metrics, f)

        module_logger.info("Testing complete: %s", scene)

    def on_test_end(self) -> None:
        if self.evaluation_cfg is not None:
            return
        name = get_cfg()["wandb"]["name"]
        out_dir = self.test_cfg.output_path / name
        if self.test_cfg.split_prefill_render:
            timing = {
                tag: list(self.benchmarker.execution_times[tag])
                for tag in ("prefill", "render", "prefill_render_total")
            }
            if self._split_metadata is None:
                raise RuntimeError("Split timing completed without metadata")
            payload = {**self._split_metadata, "seconds": timing}
            out_dir.mkdir(parents=True, exist_ok=True)
            with (out_dir / "prefill_render_benchmark.json").open("w") as handle:
                json.dump(payload, handle, indent=2)
            return
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            for metric_name, metric_scores in self.test_step_outputs.items():
                if not metric_scores:
                    raise RuntimeError(f"No values were recorded for metric {metric_name}.")
                avg_scores = float(sum(metric_scores) / len(metric_scores))
                saved_scores[metric_name] = avg_scores
                module_logger.info("%s %s", metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]) :]
                mean_time = float(np.mean(times)) if times else None
                saved_scores[tag] = [len(times), mean_time]
                module_logger.info("%s: %s calls, avg. %s seconds per call", tag, len(times), mean_time)
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / "scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)
            with (out_dir / "evaluation_summary.json").open("w") as f:
                json.dump(
                    {
                        "num_scenes": len(self.test_scenes),
                        "scenes": sorted(self.test_scenes),
                        "metric_counts": {
                            metric_name: len(metric_scores)
                            for metric_name, metric_scores in self.test_step_outputs.items()
                        },
                    },
                    f,
                    indent=2,
                )
            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
            self.benchmarker.dump_memory(
                self.test_cfg.output_path / name / "peak_memory.json"
            )
            self.benchmarker.summarize()

    def _test_evaluation_batch(self, batch: BatchedExample) -> None:
        scene = batch["scene"][0]
        if scene in self.evaluation_records:
            raise ValueError(f"Evaluation scene {scene} was produced more than once.")
        task = self.evaluation_cfg["task"]
        output, metrics = evaluate_batch(self.encoder, batch, task)
        self.evaluation_records[scene] = metrics
        scene_path = self.test_cfg.output_path / get_cfg()["wandb"]["name"] / scene
        scene_path.mkdir(parents=True, exist_ok=True)
        filename = "metrics_pose.json" if task == "pose" else "metrics.json"
        (scene_path / filename).write_text(json.dumps(metrics, indent=2, sort_keys=True))
        if task == "nvs":
            self._save_rgb_outputs(batch, output.color[0], scene_path)
            if self.test_cfg.save_video:
                self._render_eval_videos(batch, scene, scene_path)
        elif self.test_cfg.save_artifacts:
            self._save_camera_reference_files(batch, output.extrinsic, scene_path)
        module_logger.info("Evaluated %s: %s", scene, metrics)

    def finish_evaluation(self) -> None:
        # Called after Trainer.test on every rank, including ranks with no scenes.
        finalize_evaluation(
            self.evaluation_records,
            self.evaluation_cfg,
            self.test_cfg.output_path / get_cfg()["wandb"]["name"],
        )

    def on_before_optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.99).item()


        skip_optimization = (self.optimizer_cfg.grad_clip_steps >= 0) and (self.trainer.global_step > self.optimizer_cfg.grad_clip_steps) and (norms > self.optimizer_cfg.grad_clip)
        if skip_optimization or np.isnan(norms):
            module_logger.warning("Skipped optimization")
            self.zero_grad()
        self.log("train/gradient_norm", norms)
        self.log("train/ignore_step", skip_optimization * 1.0)
    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            module_logger.info(
                "validation step %s; scene=%s; context=%s",
                self.global_step,
                [a[:20] for a in batch["scene"]],
                batch["context"]["index"].tolist(),
            )

        assert batch["target"]["image"].shape[0] == 1
        output_softmax = self.encoder(
            batch["context"],
            batch["target"],
        )
        rgb_softmax = output_softmax.color[0]

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        psnr = compute_psnr(rgb_gt, rgb_softmax).mean()
        self.log("val/psnr_val", psnr)
        lpips_sum = 0
        for i in range(0, rgb_gt.shape[0], 4):
            lpips_sum += compute_lpips(rgb_gt[i:i+4], rgb_softmax[i:i+4]).sum()
        self.log("val/lpips_val", lpips_sum/rgb_gt.shape[0])
        ssim = compute_ssim(rgb_gt, rgb_softmax).mean()
        self.log("val/ssim_val", ssim)

        # Construct comparison image.
        comparison = hcat(
            add_label(vcat(*batch["context"]["image"][0]), "Context"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_softmax), "Target (Softmax)"),
        )
        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )

        if hasattr(output_softmax, "extrinsic") and output_softmax.extrinsic is not None:
            # Camera only
            scene = batch["scene"][0]
            metrics_pose = pose_metrics(output_softmax.extrinsic, batch["context"]["extrinsics"])
            module_logger.info("pose metrics: %s", metrics_pose)
            self.log(f"val/auc30", metrics_pose["Auc_30"])
            self.log(f"val/auc5",  metrics_pose["Auc_5"])
            os.makedirs(self.logger.log_path / "cameras", exist_ok=True)
            os.makedirs(self.logger.log_path / "cameras_align", exist_ok=True)

            glbscene = visualize_camera(output_softmax.extrinsic[0], batch["context"]["extrinsics"][0])
            glbscene.export(file_obj=self.logger.log_path / "cameras" / f"{self.global_step:0>6}_{scene}.ply")

            glbscene = visualize_camera(output_softmax.extrinsic[0], batch["context"]["extrinsics"][0], align=True)
            glbscene.export(file_obj=self.logger.log_path / "cameras_align" / f"{self.global_step:0>6}_{scene}.ply")

        try:
            self.render_video_interpolation(batch)
            if self.train_cfg.extended_visualization:
                self.render_video_interpolation_exaggerated(batch)
        except Exception:
            module_logger.exception("Error generating validation video")
        module_logger.info("Rank %s finished validation_step", self.global_rank)
        return


    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, v-1]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, v-1]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_all(
        self,
        batch: BatchedExample,
        save_dir,
        sample_interval: int = 1,
        use_cubic_spline: bool = False,
    ) -> None:
        _, v, _, _ = batch[self.test_cfg.extrpolation_src]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics_all = []
            intrinsics_all = []
            for view_id in range(0, v-sample_interval, sample_interval):
                extrinsics = interpolate_extrinsics(
                    batch[self.test_cfg.extrpolation_src]["extrinsics"][0, view_id],
                    (
                        batch[self.test_cfg.extrpolation_src]["extrinsics"][0, view_id+sample_interval]
                    ),
                    t,
                )
                intrinsics = interpolate_intrinsics(
                    batch[self.test_cfg.extrpolation_src]["intrinsics"][0, view_id],
                    (
                        batch[self.test_cfg.extrpolation_src]["intrinsics"][0, view_id+sample_interval]
                    ),
                    t,
                )
                extrinsics_all.append(extrinsics)
                intrinsics_all.append(intrinsics)
            extrinsics_all = torch.cat(extrinsics_all, dim=0)
            intrinsics_all = torch.cat(intrinsics_all, dim=0)
            if use_cubic_spline:
                extrinsics_all = cubicspline_interpolate(batch[self.test_cfg.extrpolation_src]["extrinsics"][0], extrinsics_all)

            return extrinsics_all[None], intrinsics_all[None]


        num_frames = 100 if v <= 4 else 10

        return self.render_video_generic(batch, trajectory_fn, "rgb", num_frames=num_frames, save_dir=save_dir)

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                batch["context"]["extrinsics"][0, 1],
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                batch["context"]["intrinsics"][0, 1],
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
        save_dir = None,
    ) -> None:
        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        num_frames = extrinsics.shape[1]

        state = self.encoder.prefill_context(batch["context"])
        chunk_size = batch["target"]["extrinsics"].shape[1]
        renderings = []
        for start in range(0, num_frames, chunk_size):
            renderings.append(self.encoder.render_targets(
                state,
                {
                    "extrinsics": extrinsics[:, start:start + chunk_size],
                    "intrinsics": intrinsics[:, start:start + chunk_size],
                },
            ))
        color = torch.cat(renderings, dim=1)
        if save_dir is not None:
            video = color[0]
        else:
            images_prob = [hcat(rgb) for rgb in color[0]]
            images = [
                add_border(
                    hcat(
                        add_label(image_prob, "Softmax"),
                    )
                )
                for image_prob in images_prob
            ]
            video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]

        if save_dir is not None:
            save_dir.mkdir(exist_ok=True, parents=True)
            frames = np.moveaxis(video, 1, -1)
            output_path = save_dir / f"{batch['scene'][0]}.mp4"
            clip = mpy.ImageSequenceClip(list(frames), fps=30)
            try:
                clip.write_videofile(
                    str(output_path),
                    codec="libx264",
                    audio=False,
                    preset="medium",
                    ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
                    logger=None,
                )
            finally:
                clip.close()
            return

        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                if save_dir is None:
                    dir = self.logger.log_path / key
                    dir.mkdir(exist_ok=True, parents=True)
                    clip.write_videofile(
                        str(dir / f"{self.global_step:0>6}_{batch['scene'][0]}.mp4"), logger=None
                    )

    def configure_optimizers(self):
        # our learning rate set configured with batch 12 and GPU 1
        scaling_rate = torch.cuda.device_count()  * int(os.environ.get('NUM_NODE', 1)) if self.optimizer_cfg.scale_lr else 1.0
        module_logger.info(
            "Scale learning rate to %s from %s by %s",
            self.optimizer_cfg.lr * scaling_rate,
            self.optimizer_cfg.lr,
            scaling_rate,
        )

        params = [p for p in self.parameters() if p.requires_grad]
        decay_params = [p for p in params if p.dim() >= 2]
        # Preserve the historical mixed-training parameter grouping.
        nodecay_params = [p for p in params if p.dim() < 2]
        filtered_param = [
            {"params": decay_params, "weight_decay": self.optimizer_cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        optimizer = optim.AdamW(filtered_param, lr=self.optimizer_cfg.lr * scaling_rate,
                               betas=(0.9, 0.95),
                               fused=True)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.optimizer_cfg.lr * scaling_rate,
            self.trainer.max_steps + 10,
            pct_start=0.01,
            cycle_momentum=False,
            anneal_strategy='cos',
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
