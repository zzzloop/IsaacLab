#!/usr/bin/env python3
"""Replay an ACT HDF5 episode through the BRX Isaac Lab control server.

This script is for imitation-replay verification, not model inference.  It reads
23-D absolute joint targets from an ACT episode, converts them to X-VLA's 20-D
absolute EE6D representation with the same lightweight URDF FK assumption used
by the training handler, then sends the trajectory to ``brx_control_server.py``.

Expected data:
  /action            [T, 23] absolute joint targets, preferred for replay
  /observations/qpos [T, 23] observed joint state, useful for diagnostics

EE6D order:
  [left_xyz, left_rot6d, left_gripper, right_xyz, right_rot6d, right_gripper]
where gripper is converted to jaw meters by default before sending to Isaac Lab.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R


EXPECTED_JOINT_NAMES = [
    "FoldingModularJoint02_Joint",
    "FoldingModularJoint03_Joint",
    "Trunk_Joint",
    "ArmL02_Joint",
    "ArmL03_Joint",
    "ArmL04_Joint",
    "ArmL05_Joint",
    "ArmL06_Joint",
    "ArmL07_Joint",
    "ArmL08_Joint",
    "JawBlock03_Joint",
    "JawBlock04_Joint",
    "ArmR02_Joint",
    "ArmR03_Joint",
    "ArmR04_Joint",
    "ArmR05_Joint",
    "ArmR06_Joint",
    "ArmR07_Joint",
    "ArmR08_Joint",
    "JawBlock01_Joint",
    "JawBlock02_Joint",
    "Head02_Joint",
    "Head03_Joint",
]
LEFT_EE_LINK = "LinearclampinggripperJZ02_Link"
RIGHT_EE_LINK = "LinearclampinggripperJZ01_Link"
DEFAULT_GRIPPER_MAX_M = 0.041


class SimpleURDFFK:
    """Minimal URDF FK for replaying the BRX ACT 23-D joint vectors."""

    def __init__(self, urdf_path: str) -> None:
        self.urdf_path = urdf_path
        root = ET.parse(urdf_path).getroot()

        self.links = [link.attrib["name"] for link in root.findall("link")]
        self.joints: list[dict[str, object]] = []
        child_links: set[str] = set()

        for joint in root.findall("joint"):
            origin = joint.find("origin")
            axis = joint.find("axis")
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            child_links.add(child)

            self.joints.append(
                {
                    "name": joint.attrib["name"],
                    "type": joint.attrib.get("type", "fixed"),
                    "parent": parent,
                    "child": child,
                    "xyz": self._parse_vec(origin.attrib.get("xyz", "0 0 0")) if origin is not None else np.zeros(3),
                    "rpy": self._parse_vec(origin.attrib.get("rpy", "0 0 0")) if origin is not None else np.zeros(3),
                    "axis": self._parse_vec(axis.attrib.get("xyz", "0 0 0")) if axis is not None else np.zeros(3),
                }
            )

        roots = [link for link in self.links if link not in child_links]
        if len(roots) != 1:
            raise ValueError(f"URDF should have exactly one root link, got {roots}")
        self.root_link = roots[0]

        # Match the old Isaac Gym active DOF order: robot_qpos[8:31] / robot_dof_state[i+8].
        self.movable_joint_names = list(EXPECTED_JOINT_NAMES)

    @staticmethod
    def _parse_vec(text: str) -> np.ndarray:
        return np.asarray([float(x) for x in text.split()], dtype=np.float64)

    @staticmethod
    def _transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = R.from_euler("xyz", rpy, degrees=False).as_matrix()
        mat[:3, 3] = xyz
        return mat

    @staticmethod
    def _motion(joint_type: str, axis: np.ndarray, q: float) -> np.ndarray:
        mat = np.eye(4, dtype=np.float64)
        norm = np.linalg.norm(axis)
        if norm > 0:
            axis = axis / norm
        if joint_type in ("revolute", "continuous"):
            mat[:3, :3] = R.from_rotvec(axis * float(q)).as_matrix()
        elif joint_type == "prismatic":
            mat[:3, 3] = axis * float(q)
        return mat

    def forward(self, q: np.ndarray, target_links: Iterable[str]) -> dict[str, np.ndarray]:
        if q.shape[-1] != len(self.movable_joint_names):
            raise ValueError(
                f"Expected q dimension {len(self.movable_joint_names)}, got {q.shape[-1]}. "
                "Check whether the ACT 23-D vector matches URDF movable joint order."
            )

        q_by_name = dict(zip(self.movable_joint_names, q.astype(np.float64).tolist()))
        target_links = set(target_links)
        link_tf = {self.root_link: np.eye(4, dtype=np.float64)}
        unresolved = list(self.joints)

        while unresolved:
            progressed = False
            rest = []
            for joint in unresolved:
                parent_tf = link_tf.get(str(joint["parent"]))
                if parent_tf is None:
                    rest.append(joint)
                    continue

                origin_tf = self._transform(joint["xyz"], joint["rpy"])
                q_value = q_by_name.get(str(joint["name"]), 0.0)
                motion_tf = self._motion(str(joint["type"]), joint["axis"], q_value)
                link_tf[str(joint["child"])] = parent_tf @ origin_tf @ motion_tf
                progressed = True

            if not progressed:
                missing = [str(joint["name"]) for joint in rest]
                raise ValueError(f"Could not resolve URDF joint tree, remaining joints: {missing}")
            unresolved = rest

        missing_links = [link for link in target_links if link not in link_tf]
        if missing_links:
            raise ValueError(f"Target EE link(s) not found in URDF tree: {missing_links}")
        return {link: link_tf[link] for link in target_links}


@dataclass
class ReplayConfig:
    gripper_max_m: float
    gripper_units: str


def rotmat_to_6d(rot: np.ndarray) -> np.ndarray:
    # 6D order is [first column xyz, second column xyz].
    return rot[:, :2].T.reshape(6)


def gripper_opening01(q: np.ndarray, pos_idx: int, neg_idx: int) -> float:
    # New BRX042501 data uses same-signed equal JawBlock values for each gripper.
    opening_m = 0.5 * (abs(float(q[pos_idx])) + abs(float(q[neg_idx])))
    return float(np.clip(opening_m / DEFAULT_GRIPPER_MAX_M, 0.0, 1.0))


def joint23_to_ee6d(q: np.ndarray, fk: SimpleURDFFK, config: ReplayConfig) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    single = q.ndim == 1
    if single:
        q = q[None]
    if q.ndim != 2 or q.shape[1] != 23:
        raise ValueError(f"Expected [T, 23] or [23], got {q.shape}")

    rows = []
    for row in q:
        poses = fk.forward(row, [LEFT_EE_LINK, RIGHT_EE_LINK])
        left_tf = poses[LEFT_EE_LINK]
        right_tf = poses[RIGHT_EE_LINK]

        # ACT slots 10/11 are the left gripper and slots 19/20 are the right gripper.
        # The BRX042501 URDF names are swapped relative to those ACT slot names:
        # left gripper = JawBlock03/04, right gripper = JawBlock01/02.
        # Keep this identical to datasets/domain_handler/custom_handler.py and brx_control_server.py.
        left_grip01 = gripper_opening01(row, 10, 11)
        right_grip01 = gripper_opening01(row, 19, 20)
        if config.gripper_units == "meters":
            left_grip = left_grip01 * config.gripper_max_m
            right_grip = right_grip01 * config.gripper_max_m
        else:
            left_grip = left_grip01
            right_grip = right_grip01

        left = np.concatenate(
            [left_tf[:3, 3], rotmat_to_6d(left_tf[:3, :3]), np.asarray([left_grip], dtype=np.float64)]
        )
        right = np.concatenate(
            [right_tf[:3, 3], rotmat_to_6d(right_tf[:3, :3]), np.asarray([right_grip], dtype=np.float64)]
        )
        rows.append(np.concatenate([left, right], axis=-1))

    out = np.stack(rows, axis=0).astype(np.float32)
    return out[0] if single else out


def post_json(url: str, payload: dict, timeout: float = 10.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed with HTTP {exc.code}: {body}") from exc


def get_json(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def print_summary(name: str, ee6d: np.ndarray) -> None:
    left_xyz = ee6d[:, 0:3]
    left_grip = ee6d[:, 9]
    right_xyz = ee6d[:, 10:13]
    right_grip = ee6d[:, 19]
    print(f"[replay] {name} shape: {ee6d.shape}")
    print(f"[replay] left xyz min/max: {np.round(left_xyz.min(axis=0), 4).tolist()} / {np.round(left_xyz.max(axis=0), 4).tolist()}")
    print(f"[replay] right xyz min/max: {np.round(right_xyz.min(axis=0), 4).tolist()} / {np.round(right_xyz.max(axis=0), 4).tolist()}")
    print(f"[replay] left grip min/max: {float(left_grip.min()):.4f} / {float(left_grip.max()):.4f}")
    print(f"[replay] right grip min/max: {float(right_grip.min()):.4f} / {float(right_grip.max()):.4f}")
    print(f"[replay] first left xyz/grip: {np.round(left_xyz[0], 4).tolist()} {float(left_grip[0]):.4f}")
    print(f"[replay] first right xyz/grip: {np.round(right_xyz[0], 4).tolist()} {float(right_grip[0]):.4f}")
    print(f"[replay] last left xyz/grip: {np.round(left_xyz[-1], 4).tolist()} {float(left_grip[-1]):.4f}")
    print(f"[replay] last right xyz/grip: {np.round(right_xyz[-1], 4).tolist()} {float(right_grip[-1]):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay ACT HDF5 action/qpos through BRX Isaac Lab server.")
    parser.add_argument("--hdf5", required=True, help="Path to ACT episode HDF5, for example /home/kemove/ACT_Datasets/episode_0.hdf5")
    parser.add_argument("--urdf_path", required=True, help="Path to BRX042501_wheel.urdf")
    parser.add_argument("--server", default="http://127.0.0.1:8765", help="BRX control server base URL")
    parser.add_argument("--dataset", choices=["action", "qpos"], default="action", help="Replay /action or /observations/qpos")
    parser.add_argument("--control_mode", choices=["ee6d", "joint23"], default="ee6d", help="ee6d uses FK+IK; joint23 sends 23-D joint targets directly")
    parser.add_argument("--start", type=int, default=0, help="Start frame, inclusive")
    parser.add_argument("--end", type=int, default=None, help="End frame, exclusive")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument("--max_rows", type=int, default=None, help="Optional maximum rows after slicing")
    parser.add_argument("--chunk_size", type=int, default=0, help="Rows per request. 0 sends the full selected trajectory once.")
    parser.add_argument("--sleep", type=float, default=0.3, help="Seconds to sleep between chunks when --chunk_size > 0")
    parser.add_argument("--stream_hz", type=float, default=0.0, help="If >0, send one row per HTTP request at this frequency, e.g. 30 for ACT 30Hz replay")
    parser.add_argument("--dry_run", action="store_true", help="Only load and convert; do not send commands")
    parser.add_argument("--stop_first", action="store_true", help="Send /command/stop before replay")
    parser.add_argument("--warmup_qpos0", type=int, default=0, help="Before replaying, send observations/qpos[0] to /command/joint23 this many times")
    parser.add_argument("--reset_qpos0", action="store_true", help="Before replaying, directly write observations/qpos[0] into the simulator via /command/reset_joint23")
    parser.add_argument("--reset_only", action="store_true", help="Only reset simulator to observations/qpos[0], then exit")
    parser.add_argument("--print_joint_table", action="store_true", help="Print per-joint first/last/min/max for the selected 23-D rows")
    parser.add_argument("--gripper_units", choices=["meters", "normalized"], default="meters", help="Server expects meters. Use normalized only for debugging raw training targets.")
    parser.add_argument("--gripper_max_m", type=float, default=DEFAULT_GRIPPER_MAX_M, help="Jaw command for fully open gripper in meters")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.chunk_size < 0:
        raise ValueError("--chunk_size must be >= 0")

    print(f"[replay] hdf5: {args.hdf5}")
    print(f"[replay] urdf: {args.urdf_path}")
    print(f"[replay] server: {args.server}")
    print(f"[replay] dataset: {args.dataset}")
    print(f"[replay] control_mode: {args.control_mode}")

    fk = SimpleURDFFK(args.urdf_path)
    print(f"[replay] URDF movable joints: {len(fk.movable_joint_names)}")
    order_ok = fk.movable_joint_names == EXPECTED_JOINT_NAMES
    print(f"[replay] movable joint order matches training assumption: {order_ok}")
    if not order_ok:
        print("[replay] URDF movable joint order:")
        for i, name in enumerate(fk.movable_joint_names):
            expected = EXPECTED_JOINT_NAMES[i] if i < len(EXPECTED_JOINT_NAMES) else "<none>"
            mark = "OK" if name == expected else "DIFF"
            print(f"  {i:02d}: {name} expected={expected} {mark}")

    key = "action" if args.dataset == "action" else "observations/qpos"
    with h5py.File(args.hdf5, "r") as f:
        if key not in f:
            raise KeyError(f"Missing dataset {key} in {args.hdf5}")
        q = np.asarray(f[key], dtype=np.float32)

    print(f"[replay] raw {key} shape: {q.shape}")
    q = q[args.start : args.end : args.stride]
    if args.max_rows is not None:
        q = q[: args.max_rows]
    if q.size == 0:
        raise ValueError("Selected trajectory is empty. Check --start/--end/--stride/--max_rows.")
    print(f"[replay] selected joint rows: {q.shape}")
    if args.print_joint_table:
        print("[replay] selected 23-D joint table:")
        for i, name in enumerate(EXPECTED_JOINT_NAMES):
            col = q[:, i]
            print(f"  {i:02d} {name}: first={float(col[0]): .5f} last={float(col[-1]): .5f} min={float(col.min()): .5f} max={float(col.max()): .5f}")

    config = ReplayConfig(gripper_max_m=args.gripper_max_m, gripper_units=args.gripper_units)
    ee6d = joint23_to_ee6d(q, fk, config)
    print_summary("converted ee6d", ee6d)

    if args.control_mode == "joint23":
        command_rows = q.astype(np.float32)
        endpoint = args.server.rstrip("/") + "/command/joint23"
        print("[replay] joint23 mode bypasses FK and IK. This is the A/B check for data order, units, and URDF joint semantics.")
        print(f"[replay] joint23 first row: {np.round(command_rows[0], 4).tolist()}")
        print(f"[replay] joint23 last row: {np.round(command_rows[-1], 4).tolist()}")
    else:
        command_rows = ee6d
        endpoint = args.server.rstrip("/") + "/command/ee6d"

    if args.dry_run:
        print("[replay] dry_run=True, not sending commands.")
        return

    state = get_json(args.server.rstrip("/") + "/health")
    print(f"[replay] server health: {state}")

    if args.stop_first:
        reply = post_json(args.server.rstrip("/") + "/command/stop", {})
        print(f"[replay] stop_first reply: {reply}")

    if args.reset_qpos0 or args.reset_only or args.warmup_qpos0 > 0:
        with h5py.File(args.hdf5, "r") as f:
            if "observations/qpos" not in f:
                raise KeyError("qpos0 reset/warmup requires observations/qpos in the HDF5 file")
            qpos0 = np.asarray(f["observations/qpos"][0], dtype=np.float32)

    if args.reset_qpos0 or args.reset_only:
        reply = post_json(args.server.rstrip("/") + "/command/reset_joint23", {"qpos": qpos0.tolist()})
        print(f"[replay] reset qpos0 reply: {reply}")
        time.sleep(0.5)
        try:
            state_after_reset = get_json(args.server.rstrip("/") + "/state")
            print(f"[replay] state after reset mode: {state_after_reset.get('mode')}")
            print(f"[replay] state qpos23 head joints: {state_after_reset.get('qpos23', [None] * 23)[21:23]}")
        except Exception as exc:
            print(f"[replay] could not read state after reset: {exc}")
        if args.reset_only:
            return

    if args.warmup_qpos0 > 0:
        warmup_rows = np.repeat(qpos0[None, :], args.warmup_qpos0, axis=0)
        reply = post_json(args.server.rstrip("/") + "/command/joint23", {"qpos": warmup_rows.tolist()})
        print(f"[replay] warmup qpos0 rows: {reply}")
        time.sleep(max(0.2, args.warmup_qpos0 * 0.02))

    payload_key = "qpos" if args.control_mode == "joint23" else "action"

    if args.stream_hz > 0:
        period = 1.0 / args.stream_hz
        total = command_rows.shape[0]
        print(f"[replay] streaming {total} rows at {args.stream_hz:.3f} Hz with payload key {payload_key}")
        next_time = time.monotonic()
        for i, row in enumerate(command_rows):
            reply = post_json(endpoint, {payload_key: [row.tolist()]})
            if i == 0 or i == total - 1 or i % max(1, int(args.stream_hz)) == 0:
                print(f"[replay] streamed row {i + 1}/{total}: {reply}")
            next_time += period
            delay = next_time - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        return

    if args.chunk_size == 0:
        reply = post_json(endpoint, {payload_key: command_rows.tolist()})
        print(f"[replay] sent all rows: {reply}")
        return

    total = command_rows.shape[0]
    for start in range(0, total, args.chunk_size):
        end = min(start + args.chunk_size, total)
        reply = post_json(endpoint, {payload_key: command_rows[start:end].tolist()})
        print(f"[replay] sent rows {start}:{end}: {reply}")
        if end < total:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
