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
parser.add_argument("--domain_id", type=int, default=0, help="Domain id passed to X-VLA server. For this custom dataset, datasets/domain_config.py maps custom -> 0.")
parser.add_argument("--steps", type=int, default=30, help="Requested X-VLA action chunk length.")
parser.add_argument("--exec_rows", type=int, default=30, help="Rows from each returned action chunk to execute.")
parser.add_argument("--cycles", type=int, default=1, help="How many observe-predict-execute cycles to run. Use -1 for forever.")
parser.add_argument("--rate_hz", type=float, default=2.0, help="Minimum bridge cycle spacing when --wait_for_chunk is disabled.")
parser.add_argument("--sim_dt", type=float, default=0.01, help="Isaac Lab simulation dt used to estimate chunk execution time.")
parser.add_argument("--command_hold_steps", type=int, default=3, help="BRX server hold steps per action row.")
parser.add_argument("--wait_for_chunk", action="store_true", default=True, help="Wait for sent rows to execute before the next observe-predict cycle.")
parser.add_argument("--no_wait_for_chunk", dest="wait_for_chunk", action="store_false", help="Do not wait for the sent action chunk to finish.")
parser.add_argument("--wait_poll_s", type=float, default=0.05, help="Polling interval while waiting for BRX to consume a command chunk.")
parser.add_argument("--wait_timeout_s", type=float, default=10.0, help="Maximum wait time for one command chunk to drain.")
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
parser.add_argument(
    "--xyz_mode",
    choices=("delta", "absolute"),
    default="delta",
    help="Interpret X-VLA xyz outputs as delta-from-current or absolute EE targets. New BRX training uses delta.",
)
parser.add_argument("--max_step_m", type=float, default=0.03, help="Max allowed xyz movement per action row, relative to previous target/current state.")
parser.add_argument("--min_z", type=float, default=0.45, help="Minimum allowed EE target z in robot base frame.")
parser.add_argument("--max_z", type=float, default=1.25, help="Maximum allowed EE target z in robot base frame.")
parser.add_argument("--gripper_min", type=float, default=0.0)
parser.add_argument("--gripper_max", type=float, default=0.041)
parser.add_argument("--normalized_gripper", action="store_true", default=True, help="Map model gripper outputs from opening [0, 1] to jaw opening meters before sending to BRX.")
parser.add_argument("--raw_gripper", dest="normalized_gripper", action="store_false", help="Use model gripper outputs as jaw opening meters directly.")
parser.add_argument("--invert_gripper", action="store_true", help="Interpret model gripper output as closedness instead of opening: sent_opening=(1-output)*gripper_max.")
parser.add_argument("--swap_grippers", action="store_true", help="Swap model left/right gripper channels before sending. Diagnostic only.")
parser.add_argument("--gripper_gamma", type=float, default=1.0, help="Sharpen normalized opening before scaling. >1 makes small openings close more decisively.")
parser.add_argument("--force_left_gripper_m", type=float, default=None, help="Override left gripper command in meters after model inference.")
parser.add_argument("--force_right_gripper_m", type=float, default=None, help="Override right gripper command in meters after model inference.")
parser.add_argument("--reject_unsafe", action="store_true", help="Reject chunks that required safety clipping instead of sending clipped commands.")
parser.add_argument("--unsafe_report_rows", type=int, default=5, help="Rows to print from safety diagnostics.")
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


def _model_action_to_ee6d(action: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    """Convert model output into absolute ee6d targets expected by /command/ee6d."""
    if args.xyz_mode == "absolute":
        return action

    current = np.asarray(state["ee6d_base"], dtype=np.float32)
    converted = action.copy()
    converted[:, 0:3] += current[0:3]
    converted[:, 10:13] += current[10:13]
    return converted



def _clip_xyz_sequence(action: np.ndarray, current: np.ndarray, offset: int, side: str) -> tuple[np.ndarray, list[str]]:
    clipped = action.copy()
    issues = []
    prev = current[offset : offset + 3].astype(np.float32).copy()
    for row_idx in range(clipped.shape[0]):
        raw = clipped[row_idx, offset : offset + 3].astype(np.float32)
        target = raw.copy()
        target[2] = np.clip(target[2], args.min_z, args.max_z)
        delta = target - prev
        dist = float(np.linalg.norm(delta))
        if dist > args.max_step_m:
            target = prev + delta / max(dist, 1e-8) * args.max_step_m
        if not np.allclose(target, raw, atol=1e-6):
            issues.append(
                f"{side}[{row_idx}] raw={raw.round(4).tolist()} safe={target.round(4).tolist()} "
                f"from={prev.round(4).tolist()} dist={dist:.4f}"
            )
        clipped[row_idx, offset : offset + 3] = target
        prev = target
    return clipped, issues


def _apply_safety_filter(action: np.ndarray, state: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    current = np.asarray(state["ee6d_base"], dtype=np.float32)
    safe = action.copy()
    issues: list[str] = []
    safe, left_issues = _clip_xyz_sequence(safe, current, 0, "left")
    safe, right_issues = _clip_xyz_sequence(safe, current, 10, "right")
    issues.extend(left_issues)
    issues.extend(right_issues)

    if args.swap_grippers:
        safe[:, [9, 19]] = safe[:, [19, 9]]
        issues.append("gripper diagnostic: swapped left/right gripper channels before scaling")

    for grip_idx, side in [(9, "left_gripper"), (19, "right_gripper")]:
        raw = safe[:, grip_idx].copy()
        if args.normalized_gripper:
            norm = np.clip(safe[:, grip_idx], 0.0, 1.0)
            if args.invert_gripper:
                norm = 1.0 - norm
            if args.gripper_gamma != 1.0:
                norm = np.power(norm, max(args.gripper_gamma, 1e-6))
            safe[:, grip_idx] = norm * args.gripper_max
        safe[:, grip_idx] = np.clip(safe[:, grip_idx], args.gripper_min, args.gripper_max)
        changed = np.where(np.abs(raw - safe[:, grip_idx]) > 1e-6)[0]
        for row_idx in changed.tolist():
            issues.append(f"{side}[{row_idx}] raw={raw[row_idx]:.4f} safe={safe[row_idx, grip_idx]:.4f}")

    if args.force_left_gripper_m is not None:
        safe[:, 9] = np.clip(float(args.force_left_gripper_m), args.gripper_min, args.gripper_max)
        issues.append(f"gripper diagnostic: forced left gripper to {safe[0, 9]:.4f} m")
    if args.force_right_gripper_m is not None:
        safe[:, 19] = np.clip(float(args.force_right_gripper_m), args.gripper_min, args.gripper_max)
        issues.append(f"gripper diagnostic: forced right gripper to {safe[0, 19]:.4f} m")

    return safe, issues


def _print_action_safety(action: np.ndarray, safe_action: np.ndarray, issues: list[str]) -> None:
    left_delta = np.linalg.norm(safe_action[:, 0:3] - action[:, 0:3], axis=1)
    right_delta = np.linalg.norm(safe_action[:, 10:13] - action[:, 10:13], axis=1)
    print("[bridge] safety clipped rows:", len(issues))
    print("[bridge] max left xyz correction:", round(float(np.max(left_delta)), 4))
    print("[bridge] max right xyz correction:", round(float(np.max(right_delta)), 4))
    for issue in issues[: max(0, args.unsafe_report_rows)]:
        print("[bridge] safety:", issue)
    if len(issues) > args.unsafe_report_rows:
        print(f"[bridge] safety: ... {len(issues) - args.unsafe_report_rows} more")


def _print_action_summary(action: np.ndarray, state: dict[str, Any]) -> None:
    current = np.asarray(state["ee6d_base"], dtype=np.float32)
    for side, offset, grip_idx in [("left", 0, 9), ("right", 10, 19)]:
        xyz = action[:, offset : offset + 3]
        step = np.linalg.norm(xyz - current[offset : offset + 3], axis=1)
        span = xyz.max(axis=0) - xyz.min(axis=0)
        print(f"[bridge] {side} last xyz/grip:", xyz[-1].round(4).tolist(), round(float(action[-1, grip_idx]), 4))
        print(f"[bridge] {side} xyz span:", span.round(4).tolist(), "max_dist_from_current:", round(float(step.max()), 4))
        print(
            f"[bridge] {side} gripper min/max:",
            round(float(action[:, grip_idx].min()), 4),
            "/",
            round(float(action[:, grip_idx].max()), 4),
        )


def _send_action_to_brx(action: np.ndarray) -> dict[str, Any]:
    rows = action[: max(1, args.exec_rows)].astype(float).tolist()
    return _post_json(_join_url(args.brx_url, "/command/ee6d"), {"action": rows})


def _wait_for_brx_chunk(response: dict[str, Any], rows_sent: int) -> None:
    version = response.get("version")
    if version is None:
        wait_s = rows_sent * max(1, args.command_hold_steps) * args.sim_dt
        print("[bridge] waiting fixed duration because response has no version:", round(wait_s, 3), "s")
        time.sleep(wait_s)
        return

    deadline = time.monotonic() + max(args.wait_timeout_s, args.wait_poll_s)
    start_t = time.monotonic()
    last_state: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        state = _get_brx_state()
        last_state = state
        state_version = state.get("command_version")
        if state_version is not None:
            state_version = int(state_version)
            # /state is updated from the simulation loop, so immediately after
            # POST /command it can lag behind the accepted command version by a
            # frame.  That is not a supersede; only a newer version means some
            # other command replaced the chunk we are waiting on.
            if state_version < int(version):
                time.sleep(max(args.wait_poll_s, 0.001))
                continue
            if state_version > int(version):
                print("[bridge] command was superseded while waiting:", state_version, ">", version)
                return
        remaining = int(state.get("command_rows_remaining", 0))
        consumed = int(state.get("command_rows_consumed", 0))
        if remaining <= 0 and consumed >= rows_sent:
            elapsed = max(time.monotonic() - start_t, 1e-6)
            print(
                "[bridge] chunk drained:",
                {
                    "version": version,
                    "consumed": consumed,
                    "remaining": remaining,
                    "wall_s": round(elapsed, 3),
                    "rows_per_s": round(consumed / elapsed, 2),
                },
            )
            return
        time.sleep(max(args.wait_poll_s, 0.001))

    if last_state is None:
        print("[bridge] wait timeout before reading BRX state")
    else:
        print(
            "[bridge] wait timeout:",
            {
                "version": version,
                "state_version": last_state.get("command_version"),
                "consumed": last_state.get("command_rows_consumed"),
                "remaining": last_state.get("command_rows_remaining"),
            },
        )


def _print_state_summary(state: dict[str, Any]) -> None:
    ee6d = np.asarray(state["ee6d_base"], dtype=np.float32)
    print("[bridge] BRX mode:", state.get("mode"))
    print("[bridge] left xyz/grip:", ee6d[0:3].round(4).tolist(), round(float(ee6d[9]), 4))
    print("[bridge] right xyz/grip:", ee6d[10:13].round(4).tolist(), round(float(ee6d[19]), 4))
    if "command_rows_remaining" in state:
        print(
            "[bridge] command queue:",
            {
                "version": state.get("command_version"),
                "consumed": state.get("command_rows_consumed"),
                "remaining": state.get("command_rows_remaining"),
            },
        )


def run_once(cycle_idx: int) -> None:
    state = _get_brx_state()
    print(f"\n[bridge] cycle {cycle_idx}")
    _print_state_summary(state)
    if args.no_execute:
        return

    model_action = _call_xvla(state)
    print("[bridge] X-VLA action shape:", tuple(model_action.shape), "xyz_mode:", args.xyz_mode)
    if args.xyz_mode == "delta":
        print("[bridge] first row left xyz_delta/grip:", model_action[0, 0:3].round(4).tolist(), round(float(model_action[0, 9]), 4))
        print("[bridge] first row right xyz_delta/grip:", model_action[0, 10:13].round(4).tolist(), round(float(model_action[0, 19]), 4))

    action = _model_action_to_ee6d(model_action, state)
    print("[bridge] first row left xyz/grip:", action[0, 0:3].round(4).tolist(), round(float(action[0, 9]), 4))
    print("[bridge] first row right xyz/grip:", action[0, 10:13].round(4).tolist(), round(float(action[0, 19]), 4))
    _print_action_summary(action, state)

    safe_action, issues = _apply_safety_filter(action, state)
    _print_action_safety(action, safe_action, issues)
    if issues and args.reject_unsafe:
        print("[bridge] reject_unsafe=True and safety clipping was required; not sending action to BRX.")
        return

    if args.dry_run:
        print("[bridge] dry_run=True, not sending action to BRX.")
        return

    response = _send_action_to_brx(safe_action)
    print("[bridge] sent to BRX:", response)
    if args.wait_for_chunk:
        rows = min(max(1, args.exec_rows), safe_action.shape[0])
        print("[bridge] waiting for BRX to consume rows:", rows)
        _wait_for_brx_chunk(response, rows)


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
        if not args.wait_for_chunk:
            time.sleep(max(0.0, 1.0 / max(args.rate_hz, 1e-6)))


if __name__ == "__main__":
    main()
