"""Validate an LVSPM evaluation index before running a benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def dataset_scene_names(root: Path, split: str) -> set[str]:
    candidates = (root / f"{split}.json", root / split / "index.json")
    for candidate in candidates:
        if candidate.is_file():
            return set(load_json_object(candidate))
    raise FileNotFoundError(
        f"Could not find {root / f'{split}.json'} or {root / split / 'index.json'}."
    )


def validate_index(
    path: Path,
    *,
    expected_contexts: int,
    expected_scenes: int,
    expected_sha256: str | None,
    dataset_root: Path | None,
    split: str,
) -> dict[str, Any]:
    actual_sha256 = sha256(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"Index checksum mismatch for {path}: expected {expected_sha256}, "
            f"got {actual_sha256}."
        )

    raw_index = load_json_object(path)
    valid_scenes: set[str] = set()
    null_scenes = 0
    target_counts: list[int] = []
    spans: list[int] = []

    for scene, entry in raw_index.items():
        if entry is None:
            null_scenes += 1
            continue
        if not isinstance(entry, dict) or set(entry) != {"context", "target"}:
            raise ValueError(
                f"Scene {scene} must contain exactly context and target lists."
            )
        context = entry["context"]
        target = entry["target"]
        if not isinstance(context, list) or not isinstance(target, list):
            raise ValueError(f"Scene {scene} context and target must be lists.")
        if len(context) != expected_contexts:
            raise ValueError(
                f"Scene {scene} has {len(context)} context views; "
                f"expected {expected_contexts}."
            )
        if not target:
            raise ValueError(f"Scene {scene} has no target views.")
        if any(type(index) is not int or index < 0 for index in context + target):
            raise ValueError(f"Scene {scene} contains a non-integer or negative index.")
        if len(set(context)) != len(context):
            raise ValueError(f"Scene {scene} has duplicate context views.")
        if len(set(target)) != len(target):
            raise ValueError(f"Scene {scene} has duplicate target views.")
        overlap = set(context).intersection(target)
        if overlap:
            raise ValueError(
                f"Scene {scene} reuses views as context and target: {sorted(overlap)}."
            )
        valid_scenes.add(scene)
        target_counts.append(len(target))
        spans.append(max(context) - min(context))

    if len(valid_scenes) != expected_scenes:
        raise ValueError(
            f"Index has {len(valid_scenes)} valid scenes; expected {expected_scenes}."
        )

    if dataset_root is not None:
        missing = valid_scenes - dataset_scene_names(dataset_root, split)
        if missing:
            sample = ", ".join(sorted(missing)[:10])
            raise ValueError(
                f"Dataset root {dataset_root} is missing {len(missing)} indexed scenes: "
                f"{sample}."
            )

    return {
        "path": str(path),
        "sha256": actual_sha256,
        "valid_scenes": len(valid_scenes),
        "null_scenes": null_scenes,
        "context_views": expected_contexts,
        "target_views_min": min(target_counts),
        "target_views_max": max(target_counts),
        "context_span_min": min(spans),
        "context_span_max": max(spans),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--expected-contexts", type=int, required=True)
    parser.add_argument("--expected-scenes", type=int, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_index(
        args.index,
        expected_contexts=args.expected_contexts,
        expected_scenes=args.expected_scenes,
        expected_sha256=args.expected_sha256,
        dataset_root=args.dataset_root,
        split=args.split,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
