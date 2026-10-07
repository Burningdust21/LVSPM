"""Pack official DL3DV Nerfstudio scenes without changing images or cameras."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tarfile
from zipfile import ZIP_STORED, ZipFile


def selected_scenes(preset, index_root):
    selected = {}
    for path in sorted((index_root / preset).glob("*.json")):
        for scene, entry in json.loads(path.read_text()).items():
            if entry is not None:
                selected.setdefault(scene, set()).update(entry["context"] + entry["target"])
    if not selected:
        raise ValueError(f"No evaluation indices found under {index_root / preset}")
    return selected


@contextmanager
def scene_reader(scene, source_root):
    tar_path = source_root / "images_tar" / f"{scene}.tar"
    if tar_path.is_file():
        with tarfile.open(tar_path) as archive:
            members = {}
            for member in archive.getmembers():
                path = Path(member.name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Unsafe archive path: {member.name}")
                if member.isdir():
                    continue
                if not member.isfile() or path in members:
                    raise ValueError(f"Expected unique regular archive files: {member.name}")
                members[path] = member
            transforms = [path for path in members if path.name == "transforms.json"]
            if len(transforms) != 1:
                raise ValueError(f"{tar_path}: expected one transforms.json")
            prefix = transforms[0].parent

            def read(relative):
                with archive.extractfile(members[prefix / relative]) as handle:
                    return handle.read()

            yield read, lambda relative: prefix / relative in members
        return
    source = source_root / scene
    if (source / "nerfstudio").is_dir():
        source = source / "nerfstudio"
    yield lambda relative: (source / relative).read_bytes(), lambda relative: (source / relative).is_file()


def pack_scene(scene, source_root, output_root, frame_indices):
    with scene_reader(scene, source_root) as (read, exists):
        return pack_scene_files(scene, output_root, frame_indices, read, exists)


def pack_scene_files(scene, output_root, frame_indices, read, exists):
    transforms_bytes = read(Path("transforms.json"))
    transforms = json.loads(transforms_bytes)
    frames = sorted(transforms["frames"], key=lambda frame: frame["file_path"])
    if not frame_indices or min(frame_indices) < 0 or max(frame_indices) >= len(frames):
        raise ValueError(f"{scene}: selected indices exceed the {len(frames)} source frames")

    images = []
    for frame in frames:
        path = Path(frame["file_path"])
        if path.parent != Path("images"):
            raise ValueError(f"{scene}: expected images/<filename>, got {path}")
        relative = Path("images_4") / path.name
        if not exists(relative):
            raise FileNotFoundError(f"{scene}/{relative}")
        images.append(relative)

    relative_zip = Path("scenes") / f"{scene}.zip"
    with ZipFile(output_root / relative_zip, "x", compression=ZIP_STORED) as archive:
        archive.writestr(f"{scene}/transforms.json", transforms_bytes)
        for relative in images:
            archive.writestr(f"{scene}/{relative.as_posix()}", read(relative))
    return relative_zip.as_posix(), {
        "frames": len(frames),
        "transforms_sha256": hashlib.sha256(transforms_bytes).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preset", choices=("dl3dv_nvs", "dl3dv_pose"), required=True)
    parser.add_argument("--scene", help="Pack one indexed scene for a local smoke test")
    parser.add_argument(
        "--index-root", type=Path,
        help="Override the packaged evaluation index directory",
    )
    args = parser.parse_args()
    if args.index_root is None:
        args.index_root = Path(__file__).resolve().parents[1] / "assets/evaluation"
    selected = selected_scenes(args.preset, args.index_root)
    if args.scene:
        selected = {args.scene: selected[args.scene]}
    args.output_root.mkdir(parents=True, exist_ok=True)
    if (args.output_root / "test.json").exists():
        raise FileExistsError(f"Choose a fresh output directory: {args.output_root}")
    (args.output_root / "scenes").mkdir(exist_ok=True)
    index, records = {}, {}
    for scene, frame_indices in sorted(selected.items()):
        index[scene], records[scene] = pack_scene(
            scene, args.input_root, args.output_root, frame_indices
        )
        print(f"{len(index)}/{len(selected)} {scene}", flush=True)
    (args.output_root / "test.json").write_text(json.dumps(index, indent=2) + "\n")
    manifest = {
        "preset": args.preset,
        "image_level": "images_4",
        "images_and_transforms_unchanged": True,
        "partial": args.scene is not None,
        "index_files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((args.index_root / args.preset).glob("*.json"))
        },
        "scenes": records,
    }
    (args.output_root / "preparation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(index)} scenes at {args.output_root}")


if __name__ == "__main__":
    main()
