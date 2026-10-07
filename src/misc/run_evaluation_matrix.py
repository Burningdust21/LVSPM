"""Run the published LVSPM equi-temporal matrix with one process per GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NVS_CHECKPOINT = Path("checkpoints/LVSPM_448x256.ckpt")
POSE_CHECKPOINT = Path("checkpoints/LVSPM_512x288.ckpt")
CHECKPOINT_SHA256 = {
    "nvs": "91e2fe4573cd6c0d61951326feba5e2012f0a50121248ecb80f52c71041a3ebd",
    "pose": "1d9b8aeca7155c90c929d42a547283f31e5b707e8793bf125ba8eb98aa20339f",
}
INDEX_PACKAGE_SHA256 = "4e6718c31d8202d60e7c22962ba36a427c25afd7b3fc14f5040dc38d6689eea0"
EVALUATION_SEED = 111123


@dataclass(frozen=True)
class EvaluationJob:
    preset: str
    views: int
    task: str

    @property
    def name(self) -> str:
        return f"{self.preset}_view{self.views}"


PUBLISHED_JOBS = (
    *(EvaluationJob("dl3dv_nvs", views, "nvs") for views in (16, 64, 128, 256)),
    *(EvaluationJob("dl3dv_pose", views, "pose") for views in (16, 32, 64, 128, 256)),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(*command: str) -> str:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()


def resolve_repo_path(path: Path, repo_root: Path) -> Path:
    """Resolve release assets relative to the repository, not the caller's CWD."""
    return (path if path.is_absolute() else repo_root / path).resolve()


def checkpoint_record(path: Path, task: str) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    checksum = file_sha256(path)
    expected = CHECKPOINT_SHA256[task]
    if checksum != expected:
        raise ValueError(
            f"{task} checkpoint checksum mismatch: expected {expected}, got {checksum}."
        )
    stat = path.stat()
    return {"path": str(path), "sha256": checksum, "bytes": stat.st_size}


def tail(path: Path, lines: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "<log unavailable>"


def job_command(python_bin: str, job: EvaluationJob, checkpoint: Path, output: Path) -> list[str]:
    return [
        python_bin, "-m", "src.main", "mode=test",
        "+evaluation=lvspm_equal_temporal",
        f"evaluation.preset={job.preset}", f"evaluation.views={job.views}",
        f"checkpointing.load={checkpoint}", f"test.output_path={output}",
    ]


def run_job(
    job: EvaluationJob,
    gpu: str,
    repo_root: Path,
    output_root: Path,
    checkpoints: dict[str, Path],
    python_bin: str,
) -> dict[str, Any]:
    job_root = output_root / job.name
    job_root.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "NUM_NODE": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(29600 + PUBLISHED_JOBS.index(job)),
            "PYTHONUNBUFFERED": "1",
            "JAXTYPING_DISABLE": "1",
        }
    )
    command = job_command(python_bin, job, checkpoints[job.task], job_root / "result")

    print(f"[gpu {gpu}] starting {job.name}", flush=True)
    with (job_root / "driver.log").open("w") as driver_log:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            text=True,
            stdout=driver_log,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{job.name} failed on GPU {gpu} with exit code {completed.returncode}.\n"
            f"{tail(job_root / 'driver.log')}"
        )

    aggregate_name = "scores_protocol_avg.json"
    aggregate_path = job_root / "result" / "lvspm" / aggregate_name
    with aggregate_path.open() as handle:
        aggregate = json.load(handle)
    print(f"[gpu {gpu}] finished {job.name}", flush=True)
    return {
        **asdict(job),
        "gpu": gpu,
        "output": str(job_root / "result"),
        "aggregate_file": aggregate_name,
        "metrics": aggregate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--nvs-checkpoint", type=Path, default=NVS_CHECKPOINT)
    parser.add_argument("--pose-checkpoint", type=Path, default=POSE_CHECKPOINT)
    parser.add_argument("--python-bin")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_root = args.output_root.resolve()
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("--gpus must contain one or more unique device identifiers.")
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse output root {output_root}.")

    python_bin = args.python_bin or os.environ.get("LVSPM_PYTHON") or sys.executable

    checkpoints = {
        "nvs": resolve_repo_path(args.nvs_checkpoint, repo_root),
        "pose": resolve_repo_path(args.pose_checkpoint, repo_root),
    }
    checkpoint_records = {
        task: checkpoint_record(path, task) for task, path in checkpoints.items()
    }
    jobs = PUBLISHED_JOBS

    output_root.mkdir(parents=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "git_commit": command_output("git", "-C", str(repo_root), "rev-parse", "HEAD"),
        "git_status": command_output("git", "-C", str(repo_root), "status", "--short"),
        "gpus": gpus,
        "gpu_inventory": command_output(
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total",
            "--format=csv,noheader",
        ),
        "checkpoints": checkpoint_records,
        "protocol": {
            "name": "lvspm_evaluation_v2",
            "config": "config/evaluation/lvspm_equal_temporal.yaml",
            "index_package_sha256": INDEX_PACKAGE_SHA256,
            "index_overrides": {
                "dl3dv_nvs_view16": {
                    "path": "assets/evaluation/dl3dv_nvs/16.json",
                    "sha256": "6efdffe3ee3349b60ca39c21c8c5cf8fcccc1cad5912f69568969a0f605da955",
                    "scenes": 140,
                },
                "dl3dv_nvs_view64": {
                    "path": "assets/evaluation/dl3dv_nvs/64.json",
                    "sha256": "bcf0e3b403e884ee092a31967f79b383a904be746693c9f2c6f3debdbc841998",
                    "scenes": 140,
                },
                "dl3dv_nvs_view128": {
                    "path": "assets/evaluation/dl3dv_nvs/128.json",
                    "sha256": "782a1a6084b14b43745618216dd787feae8d92e90aba9fb53516ecca70ed2438",
                    "scenes": 140,
                },
                "dl3dv_nvs_view256": {
                    "path": "assets/evaluation/dl3dv_nvs/256.json",
                    "sha256": "8b3fc456a34af35421d57ccbb2ddf2fd21cea75c2fc5b3d2e27f26ae67266857",
                    "scenes": 140,
                },
            },
            "seed": EVALUATION_SEED,
            "compile_model": True,
            "torch_compile_recompile_limit": 32,
            "transformer_stack_compile": "per_block_fullgraph_static",
            "attention_autocast": "bfloat16",
            "float32_matmul_precision": "high",
            "result_order": "preset_then_views",
        },
        "jobs": [asdict(job) for job in jobs],
    }
    (output_root / "matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    job_queue: queue.Queue[EvaluationJob] = queue.Queue()
    for job in jobs:
        job_queue.put(job)
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                job = job_queue.get_nowait()
            except queue.Empty:
                return
            try:
                result = run_job(
                    job,
                    gpu,
                    repo_root,
                    output_root,
                    checkpoints,
                    python_bin,
                )
                with lock:
                    results.append(result)
            except Exception as exc:
                with lock:
                    failures.append(f"{job.name}: {exc}")
            finally:
                job_queue.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    results.sort(key=lambda value: (value["preset"], value["views"]))
    (output_root / "matrix_results.json").write_text(
        json.dumps({"results": results, "failures": failures}, indent=2, sort_keys=True)
        + "\n"
    )
    if failures:
        raise RuntimeError("Evaluation failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
