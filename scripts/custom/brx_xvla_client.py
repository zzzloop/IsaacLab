# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""
Bridge X-VLA inference server to BRX Isaac Lab control server.

Expected running services:

1. BRX Isaac Lab control server:
   ./isaaclab.sh -p scripts/custom/brx_control_server.py \
       --urdf_path /home/kemove/zzk_data/IsaacLab/BRXURDF0401.urdf \
       --force_usd_conversion --no_instanceable

2. X-VLA inference server:
   python deploy.py --model_path <model_or_checkpoint_dir> --LoRA_path <optional_lora_dir> --port 8010

Then run this bridge from an environment that has numpy, requests, json_numpy, and pillow:

   python scripts/custom/brx_xvla_client.py \
       --brx_url http://127.0.0.1:8765 \
       --xvla_url http://127.0.0.1:8010/act \
       --instruction "put the block into the bucket" \
       --cycles 10

The control convention is the same as the custom dataset handler:
    ee6d_base/action = [left_xyz, left_rot6d, left_gripper, right_xyz, right_rot6d, right_gripper]
with 20 floats total, absolute in robot base frame.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import json_numpy
import numpy as np
import requests
from PIL import Image


parser = argparse.ArgumentParser(description="Bridge X-VLA /act output to BRX /command/ee6d.")
parser.add_argument("--brx_url", type=str, default="http://127.0.0.1:8765", help="BRX control server base URL.")
parser.add_argument("--xvla_url", type=str, default="http://127.0.0.1:8010/act", help="X-VLA inference endpoint URL.")
parser.add_argument("--instruction", type=str, default="put the block into the bucket")
parser.add_argument("--domain_id", type=int, default=6, help="Domain id passed to X-VLA server. Keep 6 if using RobotWin-style dual-arm ee6d.")
parser.add_argument("--steps", type=int, default=30, help="Requested X-VLA action chunk length.")
parser.add_argument("--exec_rows", type=int, default=30, help="Rows from each returned action chunk to execute.")
parser.add_argument("--cycles", type=int, default=1, help="How many observe-predict-execute cycles to run. Use -1 for forever.")
parser.add_argument("--rate_hz", type=float, default=2.0, help="Bridge cycle rate. BRX server handles per-row hold internally.")
parser.add_argument("--timeout", type=float, default=30.0)
parser.add_argument("--dry_run", action="store_true", help="Call X-VLA and print action shape, but do not send control to BRX.")
parser.add_argument("--no_execute", action="store_true", help="Only print BRX state and do not call X-VLA.")
parser.add_argument("--image_size", type=int, default=224, help="Blank/default image size.")
parser.add_argument("--image0", type=str, default=None, help="Path to image0. If absent, a blank image is used.")
parser.add_argument("--image1", type=str, default=None, help="Path to image1. If absent, image0/blank is reused.")
parser.add_argument("--image2", type=str, default=None, help="Path to image2. If absent, image0/blank is reused.")
parser.add_argument("--image0_url", type=str, default=None, help="HTTP endpoint returning image0 bytes.")
parser.add_argument("--image1_url", type=str, default=None, help="HTTP endpoint returning image1 bytes.")
parser.add_argument("--image2_url", type=str, default=None, help="HTTP endpoint returning image2 bytes.")
args = parser.parse_args()


def _join_url(base: str, path: str) -> str:
    return base.rstrip("/") + path


def _get_json(url: str) -> dict[str, Any]:
    resp = requests.get(url, timeout=args.timeout)
    resp.raise_for_status()
    return resp.json()


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    resp = requests.post(url, json=payload, timeout=args.timeout)
    resp.raise_for_status()
    return resp.json()


def _load_image_file(path: str) -> np.ndarray:
    image = Image.open(Path(path)).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _load_image_url(url: str) -> np.ndarray:
    resp = requests.get(url, timeout=args.timeout)
    resp.raise_for_status()
    image = Image.open(__import__("io").BytesIO(resp.content)).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _blank_image() -> np.ndarray:
    image = np.full((args.image_size, args.image_size, 3), 170, dtype=np.uint8)
    # Put a small color cue in the corner so a blank-image request is easy to spot in logs/records.
    image[:16, :16, :] = np.array([220, 80, 80], dtype=np.uint8)
    return image


def _resolve_images() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image0 = _load_image_url(args.image0_url) if args.image0_url else (_load_image_file(args.image0) if args.image0 else _blank_image())
    image1 = _load_image_url(args.image1_url) if args.image1_url else (_load_image_file(args.image1) if args.image1 else image0.copy())
    image2 = _load_image_url(args.image2_url) if args.image2_url else (_load_image_file(args.image2) if args.image2 else image0.copy())
    return image0, image1, image2


def _get_brx_state() -> dict[str, Any]:
    state = _get_json(_join_url(args.brx_url, "/state"))
    if not state.get("ready"):
        raise RuntimeError(f"BRX server is not ready: {state}")
    ee6d = np.asarray(state.get("ee6d_base"), dtype=np.float32)
    if ee6d.shape != (20,):
        raise RuntimeError(f"Expected BRX /state ee6d_base shape (20,), got {ee6d.shape}")
    return state


def _call_xvla(state: dict[str, Any]) -> np.ndarray:
    image0, image1, image2 = _resolve_images()
    proprio = np.asarray(state["ee6d_base"], dtype=np.float32)
    payload = {
        "domain_id": args.domain_id,
        "language_instruction": args.instruction,
        "proprio": json_numpy.dumps(proprio.tolist()),
        "image0": json_numpy.dumps(image0),
        "image1": json_numpy.dumps(image1),
        "image2": json_numpy.dumps(image2),
        "steps": args.steps,
    }
    response = _post_json(args.xvla_url, payload)
    if "action" not in response:
        raise RuntimeError(f"X-VLA response does not contain 'action': {response.keys()}")
    action = np.asarray(response["action"], dtype=np.float32)
    if action.ndim == 1:
        action = action.reshape(1, -1)
    if action.ndim != 2 or action.shape[1] != 20:
        raise RuntimeError(f"Expected X-VLA action shape [T, 20], got {action.shape}")
    if not np.all(np.isfinite(action)):
        raise RuntimeError("X-VLA action contains NaN or Inf")
    return action


def _send_action_to_brx(action: np.ndarray) -> dict[str, Any]:
    rows = action[: max(1, args.exec_rows)].astype(float).tolist()
    return _post_json(_join_url(args.brx_url, "/command/ee6d"), {"action": rows})


def _print_state_summary(state: dict[str, Any]) -> None:
    ee6d = np.asarray(state["ee6d_base"], dtype=np.float32)
    print("[bridge] BRX mode:", state.get("mode"))
    print("[bridge] left xyz/grip:", ee6d[0:3].round(4).tolist(), round(float(ee6d[9]), 4))
    print("[bridge] right xyz/grip:", ee6d[10:13].round(4).tolist(), round(float(ee6d[19]), 4))


def run_once(cycle_idx: int) -> None:
    state = _get_brx_state()
    print(f"\n[bridge] cycle {cycle_idx}")
    _print_state_summary(state)
    if args.no_execute:
        return

    action = _call_xvla(state)
    print("[bridge] X-VLA action shape:", tuple(action.shape))
    print("[bridge] first row left xyz/grip:", action[0, 0:3].round(4).tolist(), round(float(action[0, 9]), 4))
    print("[bridge] first row right xyz/grip:", action[0, 10:13].round(4).tolist(), round(float(action[0, 19]), 4))

    if args.dry_run:
        print("[bridge] dry_run=True, not sending action to BRX.")
        return

    response = _send_action_to_brx(action)
    print("[bridge] sent to BRX:", response)


def main() -> None:
    print("[bridge] BRX URL:", args.brx_url)
    print("[bridge] X-VLA URL:", args.xvla_url)
    print("[bridge] instruction:", args.instruction)
    cycle = 0
    while True:
        run_once(cycle)
        cycle += 1
        if args.cycles >= 0 and cycle >= args.cycles:
            break
        time.sleep(max(0.0, 1.0 / max(args.rate_hz, 1e-6)))


if __name__ == "__main__":
    main()