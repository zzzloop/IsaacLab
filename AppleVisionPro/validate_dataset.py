#!/usr/bin/env python3
"""Validate a BRX LeRobot v2.1/v3 dataset before training or upload."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AppleVisionPro.joint_contract import CAMERA_FEATURE_BY_NAME, CAMERA_NAMES, JOINT_NAMES_23


REQUIRED_FEATURES = (
    *(CAMERA_FEATURE_BY_NAME[name] for name in CAMERA_NAMES),
    "observation.state",
    "action",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate BRX042501 LeRobot data.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--repo_id", default=None, help="Required only when it cannot be inferred by the installed loader.")
    parser.add_argument("--skip_loader", action="store_true", help="Only validate files and metadata.")
    return parser


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    return tuple(int(item) for item in value)


def _metadata_check(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    required_dirs = [root / "data", root / "meta"]
    missing_dirs = [str(path) for path in required_dirs if not path.is_dir()]
    if missing_dirs:
        raise ValueError(f"Missing required LeRobot directories: {missing_dirs}")
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError(f"Missing canonical LeRobot metadata: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    missing_features = [key for key in REQUIRED_FEATURES if key not in features]
    if missing_features:
        raise ValueError(f"meta/info.json is missing features: {missing_features}")

    for key in ("observation.state", "action"):
        shape = _shape_tuple(features[key].get("shape"))
        if shape != (len(JOINT_NAMES_23),):
            raise ValueError(f"{key} metadata shape must be (23,), got {shape}")
        dtype = str(features[key].get("dtype", ""))
        if dtype != "float32":
            raise ValueError(f"{key} metadata dtype must be float32, got {dtype}")
        names = features[key].get("names")
        if names != {"motors": list(JOINT_NAMES_23)}:
            raise ValueError(f"{key} motor names/order do not match the BRX 23-D contract")

    camera_shapes: set[tuple[int, ...]] = set()
    for camera_name in CAMERA_NAMES:
        key = CAMERA_FEATURE_BY_NAME[camera_name]
        dtype = str(features[key].get("dtype", ""))
        if dtype not in ("video", "image"):
            raise ValueError(f"{key} must be a video/image feature, got {dtype!r}")
        shape = _shape_tuple(features[key].get("shape"))
        if len(shape) != 3 or shape[-1] != 3 or min(shape) <= 0:
            raise ValueError(f"{key} metadata shape must be [H,W,3], got {shape}")
        if features[key].get("names") != ["height", "width", "channel"]:
            raise ValueError(f"{key} names must be ['height', 'width', 'channel']")
        camera_shapes.add(shape)
    if len(camera_shapes) != 1:
        raise ValueError(f"The four camera metadata shapes must match, got {sorted(camera_shapes)}")

    video_features = [
        CAMERA_FEATURE_BY_NAME[name]
        for name in CAMERA_NAMES
        if features[CAMERA_FEATURE_BY_NAME[name]].get("dtype") == "video"
    ]
    if video_features and not (root / "videos").is_dir():
        raise ValueError(
            f"Video features are declared but the videos directory is missing: {root / 'videos'}"
        )

    parquet_files = sorted((root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise ValueError("No Parquet data shards were found")
    empty_files = [str(path) for path in parquet_files if path.stat().st_size == 0]
    if empty_files:
        raise ValueError(f"Empty Parquet files: {empty_files}")

    video_files = sorted((root / "videos").rglob("*.mp4")) if (root / "videos").is_dir() else []
    if video_features and not video_files:
        raise ValueError("Video features are declared but no MP4 files were found")

    summary = {
        "root": str(root),
        "codebase_version": info.get("codebase_version", info.get("version", "unknown")),
        "fps": info.get("fps"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "parquet_files": len(parquet_files),
        "video_files": len(video_files),
        "features": list(REQUIRED_FEATURES),
    }
    return info, summary


def _import_dataset_class() -> Any:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset
    except ImportError:
        try:
            from lerobot.datasets import LeRobotDataset

            return LeRobotDataset
        except ImportError:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

            return LeRobotDataset


def _loader_check(root: Path, repo_id: str | None) -> dict[str, Any]:
    dataset_class = _import_dataset_class()
    parameters = inspect.signature(dataset_class).parameters
    candidates: dict[str, Any] = {
        "repo_id": repo_id or root.name,
        "root": root,
        "local_files_only": True,
    }
    dataset = dataset_class(**{key: value for key, value in candidates.items() if key in parameters})
    if len(dataset) <= 0:
        raise ValueError("Official LeRobot loader opened an empty dataset")
    indices = sorted({0, len(dataset) - 1})
    checked_shapes: dict[str, list[int]] = {}
    for index in indices:
        frame = dataset[index]
        for key in REQUIRED_FEATURES:
            if key not in frame:
                raise ValueError(f"Official loader frame {index} is missing {key}")
        for key in ("observation.state", "action"):
            array = np.asarray(frame[key])
            if array.shape != (len(JOINT_NAMES_23),):
                raise ValueError(f"Loader frame {index} {key} shape is {array.shape}, expected (23,)")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Loader frame {index} {key} contains NaN/Inf")
            checked_shapes[key] = list(array.shape)
        for camera_name in CAMERA_NAMES:
            key = CAMERA_FEATURE_BY_NAME[camera_name]
            array = np.asarray(frame[key])
            if array.ndim != 3 or 3 not in array.shape:
                raise ValueError(f"Loader frame {index} {key} is not a 3-channel image: {array.shape}")
            checked_shapes[key] = list(array.shape)
    return {"loader_length": len(dataset), "checked_shapes": checked_shapes}


def main() -> None:
    args = _parser().parse_args()
    root = args.root.expanduser().resolve()
    _, summary = _metadata_check(root)
    if not args.skip_loader:
        try:
            summary.update(_loader_check(root, args.repo_id))
        except ImportError as exc:
            raise RuntimeError("LeRobot is not installed; use --skip_loader for metadata-only validation") from exc
    print(json.dumps({"ok": True, **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
