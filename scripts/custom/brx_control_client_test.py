# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""
Small HTTP client for testing scripts/custom/brx_control_server.py.

Run the Isaac Lab control server first:

    ./isaaclab.sh -p scripts/custom/brx_control_server.py \
        --urdf_path /home/kemove/zzk_data/IsaacLab/BRXURDF0401.urdf \
        --force_usd_conversion --no_instanceable

Then run this client from another terminal:

    python scripts/custom/brx_control_client_test.py --mode state
    python scripts/custom/brx_control_client_test.py --mode replay_current
    python scripts/custom/brx_control_client_test.py --mode ee6d --side right --axis x --delta 0.02
    python scripts/custom/brx_control_client_test.py --mode joint23 --joint JawBlock01_Joint --delta 0.01
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Any


EXPECTED_MOVABLE_JOINTS = [
    "FoldingModularJoint02_Joint",
    "FoldingModularJoint03_Joint",
    "Trunk_Joint",
    "ArmR02_Joint",
    "ArmR03_Joint",
    "ArmR04_Joint",
    "ArmR05_Joint",
    "ArmR06_Joint",
    "ArmR07_Joint",
    "ArmR08_Joint",
    "JawBlock01_Joint",
    "JawBlock02_Joint",
    "ArmL02_Joint",
    "ArmL03_Joint",
    "ArmL04_Joint",
    "ArmL05_Joint",
    "ArmL06_Joint",
    "ArmL07_Joint",
    "ArmL08_Joint",
    "JawBlock03_Joint",
    "JawBlock04_Joint",
    "Head02_Joint",
    "Head03_Joint",
]
AXIS_TO_OFFSET = {"x": 0, "y": 1, "z": 2}
SIDE_TO_EE6D_OFFSET = {"left": 0, "right": 10}


parser = argparse.ArgumentParser(description="Test client for BRX Isaac Lab HTTP control server.")
parser.add_argument("--server", type=str, default="http://127.0.0.1:8765", help="BRX control server base URL.")
parser.add_argument(
    "--mode",
    type=str,
    default="state",
    choices=["state", "replay_current", "ee6d", "joint23", "stop"],
    help="Test mode to run.",
)
parser.add_argument("--side", type=str, default="right", choices=["left", "right"], help="EE side for --mode ee6d.")
parser.add_argument("--axis", type=str, default="x", choices=["x", "y", "z"], help="XYZ axis for --mode ee6d.")
parser.add_argument("--delta", type=float, default=0.02, help="Small position or joint delta to apply.")
parser.add_argument(
    "--joint",
    type=str,
    default="JawBlock01_Joint",
    choices=EXPECTED_MOVABLE_JOINTS,
    help="Joint name for --mode joint23.",
)
parser.add_argument("--steps", type=int, default=1, help="Number of repeated command chunks to send.")
parser.add_argument("--sleep", type=float, default=0.25, help="Sleep seconds between repeated sends.")
args = parser.parse_args()


def _url(path: str) -> str:
    return args.server.rstrip("/") + path


def _request(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    request = urllib.request.Request(_url(path), data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            text = response.read().decode("utf-8")
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {path}: {text}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot connect to {args.server}. Is brx_control_server.py running?") from exc


def get_state() -> dict[str, Any]:
    state = _request("GET", "/state")
    if not state.get("ready"):
        raise RuntimeError(f"Server state is not ready: {state}")
    return state


def print_state_summary(state: dict[str, Any]) -> None:
    qpos23 = state.get("qpos23")
    ee6d = state.get("ee6d_base")
    print("[client] ready:", state.get("ready"))
    print("[client] mode:", state.get("mode"))
    if isinstance(qpos23, list):
        print("[client] qpos23 length:", len(qpos23))
        for idx, (name, value) in enumerate(zip(EXPECTED_MOVABLE_JOINTS, qpos23)):
            print(f"  qpos23[{idx:02d}] {name}: {value}")
    if isinstance(ee6d, list):
        print("[client] ee6d_base length:", len(ee6d))
        print("[client] left xyz/grip:", ee6d[0:3], ee6d[9])
        print("[client] right xyz/grip:", ee6d[10:13], ee6d[19])


def send_repeated(path: str, payload: dict[str, Any]) -> None:
    for step in range(max(1, args.steps)):
        response = _request("POST", path, payload)
        print(f"[client] send {path} step={step} response={response}")
        time.sleep(max(0.0, args.sleep))


def run_state() -> None:
    print_state_summary(get_state())


def run_replay_current() -> None:
    state = get_state()
    ee6d = state["ee6d_base"]
    qpos23 = state["qpos23"]
    print("[client] Replaying current joint23 once, then current ee6d once.")
    send_repeated("/command/joint23", {"qpos": qpos23})
    send_repeated("/command/ee6d", {"action": ee6d})


def run_ee6d() -> None:
    state = get_state()
    action = list(state["ee6d_base"])
    index = SIDE_TO_EE6D_OFFSET[args.side] + AXIS_TO_OFFSET[args.axis]
    before = action[index]
    action[index] = before + args.delta
    print(f"[client] ee6d {args.side}.{args.axis}: {before} -> {action[index]}")
    send_repeated("/command/ee6d", {"action": action})


def run_joint23() -> None:
    state = get_state()
    qpos = list(state["qpos23"])
    index = EXPECTED_MOVABLE_JOINTS.index(args.joint)
    before = qpos[index]
    qpos[index] = before + args.delta
    print(f"[client] joint23 {args.joint}: {before} -> {qpos[index]}")
    send_repeated("/command/joint23", {"qpos": qpos})


def run_stop() -> None:
    response = _request("POST", "/command/stop", {})
    print("[client] stop response:", response)


def main() -> None:
    if args.mode == "state":
        run_state()
    elif args.mode == "replay_current":
        run_replay_current()
    elif args.mode == "ee6d":
        run_ee6d()
    elif args.mode == "joint23":
        run_joint23()
    elif args.mode == "stop":
        run_stop()
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()