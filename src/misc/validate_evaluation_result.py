"""Validate completeness and protocol aggregation for an LVSPM evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


POSE_KEYS = {"Auc_30", "Auc_15", "Auc_5", "Auc_3", "R_ACC", "T_ACC"}
NVS_KEYS = {"psnr", "ssim", "lpips"}


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def finite_metrics(
    metrics: Any, required_keys: set[str], *, source: Path
) -> dict[str, float]:
    if not isinstance(metrics, dict) or not required_keys.issubset(metrics):
        missing = sorted(required_keys - set(metrics if isinstance(metrics, dict) else ()))
        raise ValueError(f"{source} is missing metrics: {missing}.")
    result: dict[str, float] = {}
    for key in required_keys:
        value = metrics[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{source} metric {key} is not a finite number: {value!r}.")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{source} metric {key} is not finite: {value!r}.")
        result[key] = numeric
    return result


def arithmetic_mean(
    records: list[dict[str, float]], required_keys: set[str]
) -> dict[str, float]:
    return {
        key: sum(record[key] for record in records) / len(records)
        for key in required_keys
    }


def validate_result(
    index_path: Path,
    result_dir: Path,
    task: str,
) -> dict[str, Any]:
    if task not in {"pose", "nvs"}:
        raise ValueError(f"Unsupported task: {task}.")
    index = load_json(index_path)
    if not isinstance(index, dict):
        raise ValueError(f"{index_path} must contain a JSON object.")
    expected = {scene for scene, entry in index.items() if entry is not None}
    metric_file = "metrics_pose.json" if task == "pose" else "metrics.json"
    required_keys = POSE_KEYS if task == "pose" else NVS_KEYS

    records: dict[str, dict[str, float]] = {}
    malformed: list[str] = []
    for child in result_dir.iterdir():
        if not child.is_dir():
            continue
        path = child / metric_file
        if not path.is_file():
            continue
        try:
            records[child.name] = finite_metrics(
                load_json(path), required_keys, source=path
            )
        except ValueError as exc:
            malformed.append(f"{child.name}: {exc}")

    observed = set(records)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra or malformed:
        raise ValueError(
            "Incomplete evaluation: "
            f"expected={len(expected)}, observed={len(observed)}, "
            f"missing={missing[:10]}, extra={extra[:10]}, malformed={malformed[:10]}."
        )

    aggregate_path = result_dir / "scores_all_avg.json"
    if not aggregate_path.is_file():
        raise FileNotFoundError(f"Missing aggregate {aggregate_path}.")
    stored_aggregate = finite_metrics(
        load_json(aggregate_path), required_keys, source=aggregate_path
    )

    summary_path = result_dir / "evaluation_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing evaluation summary {summary_path}.")
    summary = load_json(summary_path)
    if summary.get("num_scenes") != len(expected) or set(summary.get("scenes", [])) != expected:
        raise ValueError("Evaluation summary does not match the authoritative index.")

    scene_mean = arithmetic_mean(list(records.values()), required_keys)
    protocol_aggregate = scene_mean
    aggregation = "scene_mean"
    for key in required_keys:
        if not math.isclose(
            stored_aggregate[key], protocol_aggregate[key], rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError(
                f"Aggregate mismatch for {key}: stored={stored_aggregate[key]!r}, "
                f"recomputed={protocol_aggregate[key]!r}."
            )
    skipped_unmapped: list[str] = []

    return {
        "task": task,
        "scenes": len(observed),
        "result_dir": str(result_dir),
        "aggregation": aggregation,
        "aggregate": {
            key: protocol_aggregate[key] for key in sorted(required_keys)
        },
        "raw_scene_mean": {key: scene_mean[key] for key in sorted(required_keys)},
        "mapped_scenes": len(observed) - len(skipped_unmapped),
        "skipped_unmapped": len(skipped_unmapped),
        "skipped_unmapped_preview": skipped_unmapped[:10],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--task", choices=("pose", "nvs"), required=True)
    parser.add_argument(
        "--write-protocol-aggregate",
        action="store_true",
        help="Write the recomputed protocol aggregate to scores_protocol_avg.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_result(
        args.index,
        args.result_dir,
        args.task,
    )
    if args.write_protocol_aggregate:
        (args.result_dir / "scores_protocol_avg.json").write_text(
            json.dumps(summary["aggregate"], indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
