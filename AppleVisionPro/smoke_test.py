#!/usr/bin/env python3
"""Read-only end-to-end health check for the running BRX control server."""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AppleVisionPro.joint_contract import CAMERA_NAMES, JOINT_NAMES_23


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a live BRX four-camera server without moving the robot.")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--timeout_s", type=float, default=30.0)
    parser.add_argument("--save_dir", type=Path, default=None, help="Optionally save the four fetched PNG frames.")
    return parser


def _get(base_url: str, path: str, timeout_s: float) -> bytes:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=timeout_s) as response:
            return response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GET {path} failed: {exc}") from exc


def _json(base_url: str, path: str, timeout_s: float) -> dict[str, Any]:
    return json.loads(_get(base_url, path, timeout_s).decode("utf-8"))


def main() -> None:
    args = _parser().parse_args()
    deadline = time.monotonic() + max(args.timeout_s, 0.1)
    last_health: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            last_health = _json(args.url, "/health", min(args.timeout_s, 2.0))
            camera = last_health.get("camera", {})
            if last_health.get("ready") and tuple(camera.get("available", ())) == CAMERA_NAMES:
                break
        except RuntimeError:
            pass
        time.sleep(0.25)
    else:
        raise RuntimeError(f"BRX did not become ready with four cameras: {last_health}")

    state = _json(args.url, "/state", min(args.timeout_s, 2.0))
    if tuple(state.get("joint_names23", ())) != JOINT_NAMES_23:
        raise ValueError("The live joint_names23 contract does not match AppleVisionPro.joint_contract")
    for key in ("qpos23", "action23"):
        values = np.asarray(state.get(key), dtype=np.float64)
        if values.shape != (23,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Live {key} must be 23 finite values, got {values.shape}")

    width = int(last_health["camera"]["width"])
    height = int(last_health["camera"]["height"])
    images: dict[str, np.ndarray] = {}
    encoded: dict[str, bytes] = {}
    for name in CAMERA_NAMES:
        png = _get(args.url, f"/camera/{name}.png", min(args.timeout_s, 5.0))
        image = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
        if image.shape != (height, width, 3):
            raise ValueError(f"{name} shape is {image.shape}, expected {(height, width, 3)}")
        if float(np.std(image)) < 1.0:
            raise ValueError(f"{name} appears blank or nearly constant (std={np.std(image):.4f})")
        encoded[name] = png
        images[name] = image

    stereo_mean_abs_difference = float(
        np.mean(np.abs(images["head_left"].astype(np.int16) - images["head_right"].astype(np.int16)))
    )
    if stereo_mean_abs_difference < 0.01:
        raise ValueError("Head stereo images are effectively identical; verify EyeL/EyeR camera attachment")

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        for name, png in encoded.items():
            (args.save_dir / f"{name}.png").write_bytes(png)

    print(
        json.dumps(
            {
                "ok": True,
                "server": args.url,
                "joint_dimensions": 23,
                "camera_names": list(CAMERA_NAMES),
                "image_shape": [height, width, 3],
                "camera_frame_id": last_health["camera"]["frame_id"],
                "stereo_mean_abs_difference": round(stereo_mean_abs_difference, 4),
                "runtime": last_health.get("runtime", {}),
                "recorder": last_health.get("recorder", {}),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
