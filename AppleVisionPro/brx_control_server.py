# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""
BRX Isaac Lab control server.

Starts the BRX URDF robot in the table-top task scene and exposes local HTTP
control, four-camera, telemetry, and LeRobot recording interfaces.

- POST /command/ee6d
  Body: {"action": [20]} or {"action": [[20], ...]}
  Convention: [left_xyz(3), left_rot6d(6), left_gripper(1), right_xyz(3), right_rot6d(6), right_gripper(1)]
  The EE pose is absolute in the robot base frame, matching the X-VLA custom_handler.py FK adapter.

- POST /command/joint23
  Body: {"qpos": [23]} or {"qpos": [[23], ...]}
  Convention: absolute joint targets in the public order from
  AppleVisionPro.joint_contract.JOINT_NAMES_23.

- GET /state
  Returns current qpos in the 23D FK/training order plus left/right EE world poses.

Example:
    ./isaaclab.sh -p scripts/custom/brx_control_server.py \
        --urdf_path BRX042501/BRX042501_wheel_4cams.urdf \
        --force_usd_conversion --no_instanceable --enable_cameras --device cuda:1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="BRX Isaac Lab HTTP control server.")
parser.add_argument("--urdf_path", type=str, default="BRX042501/BRX042501_wheel_4cams.urdf")
parser.add_argument("--usd_dir", type=str, default=None)
parser.add_argument("--force_usd_conversion", action="store_true")
parser.add_argument("--no_instanceable", action="store_true")
parser.add_argument("--robot_prim", type=str, default="/World/Robot")
parser.add_argument("--left_ee_body", type=str, default="LinearclampinggripperJZ02_Link")
parser.add_argument("--right_ee_body", type=str, default="LinearclampinggripperJZ01_Link")
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--teleop_assets_root", type=str, default="teleop/assets", help="Isaac Gym teleop assets root used to mirror the data-collection scene.")
parser.add_argument("--sim_dt", type=float, default=1.0 / 30.0, help="Simulation dt. Isaac Gym data collection used 1/30 s.")
parser.add_argument("--command_hold_steps", type=int, default=1, help="Physics steps per queued command row.")
parser.add_argument("--joint_stiffness", type=float, default=2500.0, help="Position drive stiffness for all imported robot joints.")
parser.add_argument("--joint_damping", type=float, default=120.0, help="Position drive damping for all imported robot joints.")
parser.add_argument("--effort_limit", type=float, default=800.0, help="Implicit actuator effort limit.")
parser.add_argument("--velocity_limit", type=float, default=60.0, help="Implicit actuator velocity limit.")
parser.add_argument(
    "--gripper_close_m",
    type=float,
    default=0.041,
    help="Jaw joint target that corresponds to closed gripper in Isaac Lab control diagnostics.",
)
parser.add_argument(
    "--gripper_open_m",
    type=float,
    default=0.0,
    help="Jaw joint target that corresponds to open gripper in Isaac Lab control diagnostics.",
)
parser.add_argument("--no_task_scene", action="store_true")
parser.add_argument("--fixed_blocks", dest="randomize_blocks", action="store_false", help="Disable startup randomization for block colors and positions.")
parser.add_argument("--block_seed", type=int, default=None, help="Optional seed for repeatable randomized block placement.")
parser.add_argument("--camera_width", type=int, default=640)
parser.add_argument("--camera_height", type=int, default=360)
parser.add_argument(
    "--camera_update_every",
    type=int,
    default=None,
    help="Deprecated compatibility option. Overrides --camera_hz using sim_dt * N.",
)
parser.add_argument("--render_hz", type=float, default=30.0, help="WebRTC/viewport render rate.")
parser.add_argument("--camera_hz", type=float, default=15.0, help="Synchronized four-camera capture rate.")
parser.add_argument(
    "--camera_warmup_renders",
    type=int,
    default=4,
    help="Finite render-only warm-up passes before publishing the first four-camera frame.",
)
parser.add_argument("--no_realtime", dest="realtime", action="store_false", help="Disable wall-clock pacing (not recommended for teleoperation).")
parser.set_defaults(realtime=True)
parser.add_argument("--stats_interval_s", type=float, default=2.0, help="Runtime performance log interval.")
parser.add_argument("--default_head02", type=float, default=-0.17918, help="Initial/default Head02_Joint target in radians.")
parser.add_argument("--default_head03", type=float, default=-0.81304, help="Initial/default Head03_Joint target in radians.")
parser.add_argument("--enable_openxr_teleop", action="store_true", help="Read Apple Vision Pro/OpenXR hand poses in this Isaac Sim process and drive BRX directly.")
parser.add_argument("--openxr_scale", type=float, default=1.0, help="OpenXR hand delta to robot EE delta scale.")
parser.add_argument("--openxr_axis_map", type=str, default="x,y,z", help="Map OpenXR delta axes into robot base axes, e.g. 'z,x,y'.")
parser.add_argument("--openxr_rate_hz", type=float, default=30.0, help="Maximum OpenXR teleop command rate.")
parser.add_argument("--openxr_max_step_m", type=float, default=0.05, help="Maximum EE target movement per OpenXR update.")
parser.add_argument("--openxr_min_z", type=float, default=0.35)
parser.add_argument("--openxr_max_z", type=float, default=1.35)
parser.add_argument("--openxr_pinch_close_m", type=float, default=0.025, help="Thumb-index tip distance treated as closed gripper.")
parser.add_argument("--openxr_pinch_open_m", type=float, default=0.085, help="Thumb-index tip distance treated as open gripper.")
parser.add_argument("--record_root", type=str, default=None, help="Local LeRobot dataset root. Omit to disable recording.")
parser.add_argument("--record_repo_id", type=str, default="local/brx042501_visionpro")
parser.add_argument("--record_fps", type=int, default=None, help="Dataset FPS; defaults to rounded camera_hz.")
parser.add_argument("--record_task", type=str, default=None, help="If set with --record_root, start the first episode automatically.")
parser.add_argument("--record_queue_size", type=int, default=64)
parser.add_argument("--record_images", dest="record_use_videos", action="store_false", help="Store image features instead of MP4 video features.")
parser.set_defaults(record_use_videos=True)
parser.add_argument("--lerobot_format", choices=["auto", "v2", "v3"], default="auto")
parser.add_argument(
    "--camera_pose_mode",
    choices=["link", "lookat"],
    default="link",
    help="link matches Isaac Gym set_camera_transform(pos, quat); lookat keeps a forward/down wrist view for policy debugging.",
)
parser.add_argument(
    "--head_camera_offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    help="Head camera local xyz offset in head_camera_body frame.",
)
parser.add_argument(
    "--head_camera_forward",
    type=float,
    nargs=3,
    default=(0.25, 0.0, 0.0),
    help="Local look-at vector for /camera/head.png in head_camera_body frame.",
)
parser.add_argument(
    "--wrist_camera_forward",
    type=float,
    nargs=3,
    default=(0.20, 0.0, -0.12),
    help="Local look-at vector for wrist cameras in their URDF camera body frames.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import torch.nn.functional as F
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix, subtract_frame_transforms

from AppleVisionPro.camera_manager import BrxCameraConfig, BrxCameraManager, CameraFrameBuffer
from AppleVisionPro.joint_contract import (
    CAMERA_NAMES,
    JOINT_NAMES_23,
    LEFT_ARM_JOINTS,
    LEFT_GRIPPER_JOINTS,
    RIGHT_ARM_JOINTS,
    RIGHT_GRIPPER_JOINTS,
    inspect_urdf_contract,
    require_vector23,
    resolve_joint_ids,
)
from AppleVisionPro.lerobot_recorder import (
    LeRobotRecorder,
    RecorderConfig,
    SynchronizedSample,
    disabled_recorder_status,
)

try:
    from isaaclab.devices import DeviceBase, OpenXRDevice, OpenXRDeviceCfg, RetargeterBase
    from isaaclab.devices.openxr import XrCfg
except Exception:
    DeviceBase = None
    OpenXRDevice = None
    OpenXRDeviceCfg = None
    RetargeterBase = None
    XrCfg = None


@dataclass(frozen=True)
class ArmIkContext:
    entity_cfg: SceneEntityCfg
    controller: DifferentialIKController
    jacobian_body_index: int


class CommandBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.mode: str | None = None
        self.queue: list[list[float]] = []
        self.last: list[float] | None = None
        self.version = 0
        self.rows_total = 0
        self.rows_consumed = 0
        self.state: dict[str, Any] = {"ready": False}

    def set_command(self, mode: str, rows: list[list[float]]) -> int:
        with self._lock:
            self.mode = mode
            self.queue = [list(row) for row in rows]
            self.last = None
            self.version += 1
            self.rows_total = len(rows)
            self.rows_consumed = 0
            return self.version

    def next_row(self) -> tuple[str | None, list[float] | None, int]:
        with self._lock:
            if self.mode is None:
                return None, None, self.version
            if self.queue:
                self.last = self.queue.pop(0)
                self.rows_consumed += 1
            return self.mode, self.last, self.version

    def command_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "command_version": self.version,
                "command_mode": self.mode,
                "command_rows_total": self.rows_total,
                "command_rows_consumed": self.rows_consumed,
                "command_rows_remaining": len(self.queue),
            }

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.state)

    def set_state(self, state: dict[str, Any]) -> None:
        with self._lock:
            self.state = state


class RuntimeMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started_t = time.monotonic()
        self.physics_steps = 0
        self.render_frames = 0
        self.camera_frames = 0
        self.pacing_resets = 0
        self.command_errors = 0
        self.last_command_error: str | None = None
        self._last_report_t = self.started_t
        self._last_counts = (0, 0, 0)
        self._status: dict[str, Any] = {
            "physics_fps": 0.0,
            "render_fps": 0.0,
            "camera_fps": 0.0,
            "real_time_factor": 0.0,
        }

    def physics(self) -> None:
        with self._lock:
            self.physics_steps += 1

    def render(self) -> None:
        with self._lock:
            self.render_frames += 1

    def camera(self) -> None:
        with self._lock:
            self.camera_frames += 1

    def pacing_reset(self) -> None:
        with self._lock:
            self.pacing_resets += 1

    def command_error(self, error: BaseException) -> None:
        with self._lock:
            self.command_errors += 1
            self.last_command_error = str(error)

    def maybe_report(self, sim_dt: float, interval_s: float) -> None:
        now = time.monotonic()
        with self._lock:
            elapsed = now - self._last_report_t
            if elapsed < max(interval_s, 0.1):
                return
            previous_physics, previous_render, previous_camera = self._last_counts
            delta_physics = self.physics_steps - previous_physics
            delta_render = self.render_frames - previous_render
            delta_camera = self.camera_frames - previous_camera
            self._status = {
                "physics_fps": delta_physics / elapsed,
                "render_fps": delta_render / elapsed,
                "camera_fps": delta_camera / elapsed,
                "real_time_factor": delta_physics * sim_dt / elapsed,
            }
            self._last_report_t = now
            self._last_counts = (self.physics_steps, self.render_frames, self.camera_frames)
            status = dict(self._status)
        print(
            "[BRX][perf] "
            f"physics={status['physics_fps']:.1f}Hz render={status['render_fps']:.1f}Hz "
            f"camera={status['camera_fps']:.1f}Hz rtf={status['real_time_factor']:.3f}"
        )

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                **{key: round(float(value), 4) for key, value in self._status.items()},
                "physics_steps": self.physics_steps,
                "render_frames": self.render_frames,
                "camera_frames": self.camera_frames,
                "pacing_resets": self.pacing_resets,
                "command_errors": self.command_errors,
                "last_command_error": self.last_command_error,
                "uptime_s": round(time.monotonic() - self.started_t, 3),
            }


COMMAND_BUFFER = CommandBuffer()
RUNTIME_METRICS = RuntimeMetrics()
CAMERA_BUFFER: CameraFrameBuffer | None = None
RECORDER: LeRobotRecorder | None = None


def _camera_buffer() -> CameraFrameBuffer:
    if CAMERA_BUFFER is None:
        raise RuntimeError("Camera manager is not initialized")
    return CAMERA_BUFFER


def _recorder_status() -> dict[str, Any]:
    return disabled_recorder_status() if RECORDER is None else RECORDER.status()


def _parse_axis_map(spec: str) -> list[tuple[int, float]]:
    axes = {"x": 0, "y": 1, "z": 2}
    mapping: list[tuple[int, float]] = []
    for raw in spec.split(","):
        token = raw.strip().lower()
        sign = -1.0 if token.startswith("-") else 1.0
        token = token[1:] if token.startswith("-") else token
        if token not in axes:
            raise ValueError(f"Bad axis map token: {raw!r}")
        mapping.append((axes[token], sign))
    if len(mapping) != 3:
        raise ValueError("Axis map must contain exactly 3 comma-separated axes")
    return mapping


def _map_axis_delta(delta: np.ndarray, mapping: list[tuple[int, float]]) -> np.ndarray:
    return np.asarray([sign * delta[idx] for idx, sign in mapping], dtype=np.float32)


def _clip_xyz_np(target: np.ndarray, previous: np.ndarray, min_z: float, max_z: float, max_step: float) -> np.ndarray:
    out = target.astype(np.float32).copy()
    out[2] = np.clip(out[2], min_z, max_z)
    delta = out - previous
    dist = float(np.linalg.norm(delta))
    if dist > max_step:
        out = previous + delta / max(dist, 1e-8) * max_step
    return out


@dataclass
class OpenXRCalibration:
    left_wrist: np.ndarray
    right_wrist: np.ndarray
    left_ee: np.ndarray
    right_ee: np.ndarray


class OpenXRTeleop:
    """Use Isaac Lab OpenXR hand tracking as BRX ee6d setpoints."""

    def __init__(self) -> None:
        if OpenXRDevice is None or OpenXRDeviceCfg is None or XrCfg is None or RetargeterBase is None:
            raise RuntimeError("Isaac Lab OpenXR device is unavailable in this environment")
        self.device = OpenXRDevice(OpenXRDeviceCfg(xr_cfg=XrCfg()))
        self.device._required_features = {RetargeterBase.Requirement.HAND_TRACKING, RetargeterBase.Requirement.HEAD_TRACKING}
        self.axis_map = _parse_axis_map(args_cli.openxr_axis_map)
        self.calib: OpenXRCalibration | None = None
        self.last_t = 0.0
        print("[BRX][OpenXR] enabled. Start Isaac Lab with --xr and connect Apple Vision Pro/CloudXR.")
        print(f"[BRX][OpenXR] axis_map={args_cli.openxr_axis_map}, scale={args_cli.openxr_scale}")

    def close(self) -> None:
        pass

    def _hand(self, data: dict[Any, Any], target: Any) -> dict[str, np.ndarray] | None:
        if target in data:
            return data[target]
        for key, value in data.items():
            if getattr(key, "name", None) == getattr(target, "name", None):
                return value
        return None

    def _grip_from_hand(self, hand: dict[str, np.ndarray]) -> float:
        thumb = np.asarray(hand.get("thumb_tip", hand["wrist"])[0:3], dtype=np.float32)
        index = np.asarray(hand.get("index_tip", hand["wrist"])[0:3], dtype=np.float32)
        dist = float(np.linalg.norm(thumb - index))
        denom = max(args_cli.openxr_pinch_open_m - args_cli.openxr_pinch_close_m, 1e-6)
        open_ratio = np.clip((dist - args_cli.openxr_pinch_close_m) / denom, 0.0, 1.0)
        # BRX jaw q=0 is open and q=0.041 m is closed.
        closed_ratio = 1.0 - open_ratio
        return float(args_cli.gripper_open_m + closed_ratio * (args_cli.gripper_close_m - args_cli.gripper_open_m))

    def advance(self, state: dict[str, Any]) -> list[float] | None:
        now = time.monotonic()
        if now - self.last_t < 1.0 / max(args_cli.openxr_rate_hz, 1e-6):
            return None
        data = self.device.advance()
        if DeviceBase is None:
            return None
        left = self._hand(data, DeviceBase.TrackingTarget.HAND_LEFT)
        right = self._hand(data, DeviceBase.TrackingTarget.HAND_RIGHT)
        if not left or not right or "wrist" not in left or "wrist" not in right:
            return None

        ee6d = np.asarray(state["ee6d_base"], dtype=np.float32)
        left_wrist = np.asarray(left["wrist"][0:3], dtype=np.float32)
        right_wrist = np.asarray(right["wrist"][0:3], dtype=np.float32)
        if self.calib is None:
            self.calib = OpenXRCalibration(left_wrist=left_wrist.copy(), right_wrist=right_wrist.copy(), left_ee=ee6d[0:3].copy(), right_ee=ee6d[10:13].copy())
            print("[BRX][OpenXR] calibrated from current wrist and BRX EE poses")

        assert self.calib is not None
        row = ee6d.copy()
        left_target = self.calib.left_ee + args_cli.openxr_scale * _map_axis_delta(left_wrist - self.calib.left_wrist, self.axis_map)
        right_target = self.calib.right_ee + args_cli.openxr_scale * _map_axis_delta(right_wrist - self.calib.right_wrist, self.axis_map)
        row[0:3] = _clip_xyz_np(left_target, ee6d[0:3], args_cli.openxr_min_z, args_cli.openxr_max_z, args_cli.openxr_max_step_m)
        row[10:13] = _clip_xyz_np(right_target, ee6d[10:13], args_cli.openxr_min_z, args_cli.openxr_max_z, args_cli.openxr_max_step_m)
        row[9] = self._grip_from_hand(left)
        row[19] = self._grip_from_hand(right)
        self.last_t = now
        return row.astype(float).tolist()


def _abs_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _asset_path(*parts: str) -> str:
    root = args_cli.teleop_assets_root
    if not os.path.isabs(root):
        root = os.path.join(os.getcwd(), root)
    return _abs_path(os.path.join(root, *parts))


def _rows_from_payload(payload: dict[str, Any], key: str, width: int) -> list[list[float]]:
    if key not in payload:
        raise ValueError(f"Missing JSON field: {key}")
    value = payload[key]
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    candidate_rows = [value] if len(value) == width and all(isinstance(x, (int, float)) for x in value) else value
    rows: list[list[float]] = []
    for row in candidate_rows:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError(f"Each {key} row must have length {width}")
        values = [float(x) for x in row]
        if not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
            raise ValueError(f"Each {key} row must contain only finite values")
        rows.append(values)
    if not rows:
        raise ValueError(f"{key} cannot be empty")
    return rows


def _validate_ee6d_rows(rows: list[list[float]]) -> None:
    for row_index, row in enumerate(rows):
        for label, offset in (("left", 3), ("right", 13)):
            first = np.asarray(row[offset : offset + 3], dtype=np.float64)
            second = np.asarray(row[offset + 3 : offset + 6], dtype=np.float64)
            if np.linalg.norm(first) < 1e-6:
                raise ValueError(f"ee6d row {row_index} {label} rot6d first axis is degenerate")
            first /= np.linalg.norm(first)
            second_orthogonal = second - np.dot(first, second) * first
            if np.linalg.norm(second_orthogonal) < 1e-6:
                raise ValueError(f"ee6d row {row_index} {label} rot6d second axis is degenerate")


class ControlHandler(BaseHTTPRequestHandler):
    server_version = "BRXControlHTTP/0.2"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[HTTP] {self.address_string()} - {fmt % args}")

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_png(self, code: int, data: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        if self.path == "/state":
            self._send_json(200, COMMAND_BUFFER.get_state())
        elif self.path == "/health":
            self._send_json(
                200,
                {
                    "ok": True,
                    "ready": COMMAND_BUFFER.get_state().get("ready", False),
                    "runtime": RUNTIME_METRICS.status(),
                    "camera": _camera_buffer().status(),
                    "recorder": _recorder_status(),
                },
            )
        elif self.path == "/camera":
            self._send_json(200, _camera_buffer().status())
        elif self.path == "/record/status":
            self._send_json(200, _recorder_status())
        elif self.path in (
            "/camera/head.png",
            "/camera/head_left.png",
            "/camera/head_right.png",
            "/camera/left_wrist.png",
            "/camera/right_wrist.png",
        ):
            name = self.path.split("/")[-1].replace(".png", "")
            if name == "head":
                name = "head_left"
            frame = _camera_buffer().get_png(name)
            if frame is None:
                self._send_json(503, {"error": f"camera frame not ready: {name}"})
            else:
                self._send_png(200, frame)
        else:
            self._send_json(404, {"error": "unknown endpoint"})

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
            if self.path == "/command/ee6d":
                rows = _rows_from_payload(payload, "action", 20)
                _validate_ee6d_rows(rows)
                version = COMMAND_BUFFER.set_command("ee6d", rows)
                self._send_json(200, {"ok": True, "mode": "ee6d", "rows": len(rows), "version": version})
            elif self.path == "/command/joint23":
                rows = _rows_from_payload(payload, "qpos", 23)
                version = COMMAND_BUFFER.set_command("joint23", rows)
                self._send_json(200, {"ok": True, "mode": "joint23", "rows": len(rows), "version": version})
            elif self.path == "/command/reset_joint23":
                rows = _rows_from_payload(payload, "qpos", 23)
                if len(rows) != 1:
                    raise ValueError("reset_joint23 expects exactly one 23D qpos row")
                version = COMMAND_BUFFER.set_command("reset_joint23", rows)
                self._send_json(200, {"ok": True, "mode": "reset_joint23", "rows": len(rows), "version": version})
            elif self.path == "/command/stop":
                version = COMMAND_BUFFER.set_command("stop", [])
                self._send_json(200, {"ok": True, "mode": "stop", "version": version})
            elif self.path == "/command/gripper":
                left = payload.get("left", None)
                right = payload.get("right", None)
                if left is not None and not np.isfinite(float(left)):
                    raise ValueError("left gripper target must be finite")
                if right is not None and not np.isfinite(float(right)):
                    raise ValueError("right gripper target must be finite")
                state = COMMAND_BUFFER.get_state()
                if "ee6d_base" not in state:
                    raise ValueError("BRX state is not ready")
                row = list(state["ee6d_base"])
                if left is not None:
                    row[9] = float(left)
                if right is not None:
                    row[19] = float(right)
                row_count = int(payload.get("rows", 30))
                if not 1 <= row_count <= 10_000:
                    raise ValueError("gripper rows must be between 1 and 10000")
                rows = [row for _ in range(row_count)]
                version = COMMAND_BUFFER.set_command("ee6d", rows)
                self._send_json(200, {"ok": True, "mode": "ee6d", "rows": len(rows), "version": version, "left": row[9], "right": row[19]})
            elif self.path == "/record/start":
                if RECORDER is None:
                    raise RuntimeError("Recording is disabled; restart with --record_root")
                task = payload.get("task", args_cli.record_task)
                status = RECORDER.start_episode(str(task or ""))
                self._send_json(200, {"ok": True, "recorder": status})
            elif self.path == "/record/stop":
                if RECORDER is None:
                    raise RuntimeError("Recording is disabled; restart with --record_root")
                status = RECORDER.stop_episode(save=True)
                self._send_json(200, {"ok": True, "recorder": status})
            elif self.path == "/record/abort":
                if RECORDER is None:
                    raise RuntimeError("Recording is disabled; restart with --record_root")
                status = RECORDER.stop_episode(save=False)
                self._send_json(200, {"ok": True, "recorder": status})
            elif self.path == "/record/finalize":
                if RECORDER is None:
                    raise RuntimeError("Recording is disabled; restart with --record_root")
                RECORDER.finalize()
                self._send_json(200, {"ok": True, "recorder": RECORDER.status()})
            else:
                self._send_json(404, {"error": "unknown endpoint"})
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})


def _start_http_server() -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((args_cli.host, args_cli.port), ControlHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[BRX] HTTP control server listening on http://{args_cli.host}:{args_cli.port}")
    return server


def _make_robot_cfg() -> ArticulationCfg:
    urdf_path = _abs_path(args_cli.urdf_path)
    if not check_file_path(urdf_path):
        raise FileNotFoundError(f"URDF path does not exist or is not readable: {urdf_path}")
    report = inspect_urdf_contract(urdf_path)
    print(
        f"[BRX] URDF contract OK: links={report.link_count}, joints={report.joint_count}, "
        f"non_fixed={report.non_fixed_joint_count}, stereo_baseline={report.head_stereo_baseline_m:.3f}m"
    )
    usd_dir = _abs_path(args_cli.usd_dir) if args_cli.usd_dir else os.path.join(os.path.dirname(urdf_path), "isaaclab_converted")
    return ArticulationCfg(
        prim_path=args_cli.robot_prim,
        spawn=sim_utils.UrdfFileCfg(
            asset_path=urdf_path,
            usd_dir=usd_dir,
            usd_file_name=f"{os.path.splitext(os.path.basename(urdf_path))[0]}_imported.usd",
            force_usd_conversion=args_cli.force_usd_conversion,
            make_instanceable=not args_cli.no_instanceable,
            fix_base=True,
            merge_fixed_joints=False,
            self_collision=False,
            collision_from_visuals=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=args_cli.joint_stiffness, damping=args_cli.joint_damping),
                target_type="position",
                drive_type="force",
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        # Isaac Gym data collection spawned the robot at (-1.1, 0, 1.6).
        # Keeping the same world pose makes camera/world renderings match the dataset scene.
        init_state=ArticulationCfg.InitialStateCfg(pos=(-1.1, 0.0, 1.6)),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=args_cli.effort_limit,
                velocity_limit_sim=args_cli.velocity_limit,
                stiffness=800.0,
                damping=40.0,
            )
        },
    )


def _make_material(color: tuple[float, float, float], roughness: float = 0.7) -> sim_utils.PreviewSurfaceCfg:
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=roughness)


def _spawn_static_cuboid(path: str, size: tuple[float, float, float], pos: tuple[float, float, float], color: tuple[float, float, float]) -> None:
    cfg = sim_utils.CuboidCfg(size=size, collision_props=sim_utils.CollisionPropertiesCfg(), visual_material=_make_material(color))
    cfg.func(path, cfg, translation=pos)


def _spawn_rigid_cube(path: str, size: float, pos: tuple[float, float, float], color: tuple[float, float, float]) -> None:
    cfg = sim_utils.CuboidCfg(
        size=(size, size, size),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=8, solver_velocity_iteration_count=0),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.1),
        visual_material=_make_material(color),
    )
    cfg.func(path, cfg, translation=pos)


def _spawn_floor_visuals() -> None:
    sim_utils.create_prim("/World/FloorVisuals", "Xform")
    floor_cfg = sim_utils.CuboidCfg(size=(4.0, 4.0, 0.004), visual_material=_make_material((0.72, 0.76, 0.76), 0.9))
    floor_cfg.func("/World/FloorVisuals/Base", floor_cfg, translation=(0.35, 0.0, -0.003))
    for idx in range(-8, 9):
        offset = idx * 0.25
        line_x = sim_utils.CuboidCfg(size=(0.006, 4.0, 0.002), visual_material=_make_material((0.50, 0.55, 0.55), 0.95))
        line_x.func(f"/World/FloorVisuals/GridX_{idx + 8:02d}", line_x, translation=(0.35 + offset, 0.0, 0.001))
        line_y = sim_utils.CuboidCfg(size=(4.0, 0.006, 0.002), visual_material=_make_material((0.50, 0.55, 0.55), 0.95))
        line_y.func(f"/World/FloorVisuals/GridY_{idx + 8:02d}", line_y, translation=(0.35, offset, 0.001))


def _spawn_bucket(prefix: str, center: tuple[float, float, float]) -> None:
    x, y, table_z = center
    wall_t, outer, height, bottom_t = 0.018, 0.20, 0.16, 0.018
    base_z = table_z + bottom_t * 0.5
    wall_z = table_z + bottom_t + height * 0.5
    color = (0.95, 0.72, 0.18)
    _spawn_static_cuboid(f"{prefix}/Bottom", (outer, outer, bottom_t), (x, y, base_z), color)
    _spawn_static_cuboid(f"{prefix}/WallPosX", (wall_t, outer, height), (x + outer * 0.5, y, wall_z), color)
    _spawn_static_cuboid(f"{prefix}/WallNegX", (wall_t, outer, height), (x - outer * 0.5, y, wall_z), color)
    _spawn_static_cuboid(f"{prefix}/WallPosY", (outer, wall_t, height), (x, y + outer * 0.5, wall_z), color)
    _spawn_static_cuboid(f"{prefix}/WallNegY", (outer, wall_t, height), (x, y - outer * 0.5, wall_z), color)


def _spawn_teleop_bucket(path: str, pos: tuple[float, float, float]) -> None:
    bucket_urdf = _asset_path("bucket", "bucket.urdf")
    if os.path.exists(bucket_urdf):
        cfg = sim_utils.UrdfFileCfg(
            asset_path=bucket_urdf,
            fix_base=True,
            visual_material=_make_material((0.7, 0.7, 1.0)),
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
                target_type="position",
                drive_type="force",
            ),
        )
        cfg.func(path, cfg, translation=pos)
    else:
        print(f"[BRX] teleop bucket asset not found, using cuboid fallback: {bucket_urdf}")
        _spawn_bucket(path, (pos[0], pos[1], pos[2]))


def _random_block_layout() -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    rng = random.Random(args_cli.block_seed)
    colors = [(rng.random(), rng.random(), rng.random()), (rng.random(), rng.random(), rng.random())]
    # Isaac Gym collection scene, in world coordinates.
    positions = [
        (-0.55 + rng.uniform(-0.10, 0.10), rng.uniform(-0.05, 0.05), 2.30),
        (-0.55 + rng.uniform(-0.10, 0.10), 0.20 + rng.uniform(-0.05, 0.05), 2.30),
    ]
    return [
        ("BlockA", positions[0], colors[0]),
        ("BlockB", positions[1], colors[1]),
    ]


def _spawn_scene() -> None:
    ground = sim_utils.GroundPlaneCfg(
        color=(0.5, 0.5, 0.5),
        size=(100.0, 100.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.2, dynamic_friction=0.2, restitution=0.0),
    )
    ground.func("/World/defaultGroundPlane", ground)
    # Isaac Gym collection used the default ground and did not add decorative floor geometry.
    light = sim_utils.DomeLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0))
    light.func("/World/Light", light)
    if args_cli.no_task_scene:
        return
    sim_utils.create_prim("/World/TaskScene", "Xform")
    _spawn_static_cuboid("/World/TaskScene/TableTop", (0.8, 0.8, 0.1), (-0.30, 0.0, 2.15), (0.5, 0.5, 0.5))
    _spawn_teleop_bucket("/World/TaskScene/Bucket", (-0.30, 0.0, 2.20))
    cube_size = 0.05
    if args_cli.randomize_blocks:
        blocks = _random_block_layout()
    else:
        blocks = [
            ("BlockA", (-0.55, 0.0, 2.30), (1.0, 0.5, 0.5)),
            ("BlockB", (-0.55, 0.20, 2.30), (1.0, 0.5, 0.5)),
        ]
    for name, pos, color in blocks:
        _spawn_rigid_cube(f"/World/TaskScene/{name}", cube_size, pos, color)
        print(f"[BRX] spawned {name}: pos={tuple(round(v, 4) for v in pos)}, color={tuple(round(v, 3) for v in color)}")



def _resolve_arm(sim: SimulationContext, robot: Articulation, joint_names: list[str], body_name: str) -> ArmIkContext:
    entity_cfg = SceneEntityCfg("robot", joint_names=joint_names, body_names=[body_name])
    entity_cfg.resolve({"robot": robot})
    controller = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=1,
        device=sim.device,
    )
    jacobian_body_index = entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else entity_cfg.body_ids[0]
    return ArmIkContext(entity_cfg=entity_cfg, controller=controller, jacobian_body_index=jacobian_body_index)


def _rot6d_to_quat(rot6d: torch.Tensor) -> torch.Tensor:
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    rot_mat = torch.stack((b1, b2, b3), dim=-1)
    return quat_from_matrix(rot_mat)


def _quat_to_rot6d(quat: torch.Tensor) -> list[float]:
    mat = matrix_from_quat(quat.reshape(1, 4))[0]
    # 6D order is [first column xyz, second column xyz], matching _rot6d_to_quat().
    return mat[:, 0:2].T.reshape(-1).detach().cpu().tolist()


def _ee10_to_pose(row10: list[float], device: str) -> tuple[torch.Tensor, float]:
    values = torch.tensor(row10, dtype=torch.float32, device=device)
    pos = values[0:3]
    quat = _rot6d_to_quat(values[3:9].reshape(1, 6))[0]
    grip = float(values[9].detach().cpu())
    return torch.cat([pos, quat], dim=0).reshape(1, 7), grip


def _compute_arm_command(robot: Articulation, ctx: ArmIkContext) -> torch.Tensor:
    jacobian = robot.root_physx_view.get_jacobians()[:, ctx.jacobian_body_index, :, ctx.entity_cfg.joint_ids]
    ee_pose_w = robot.data.body_pose_w[:, ctx.entity_cfg.body_ids[0]]
    root_pose_w = robot.data.root_pose_w
    joint_pos = robot.data.joint_pos[:, ctx.entity_cfg.joint_ids]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
    return ctx.controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)


def _clamp_controlled_target(robot: Articulation, target: torch.Tensor, joint23_ids: list[int]) -> torch.Tensor:
    target = target.clone()
    limits = robot.data.soft_joint_pos_limits[0, joint23_ids]
    values = target[:, joint23_ids]
    target[:, joint23_ids] = torch.maximum(torch.minimum(values, limits[:, 1]), limits[:, 0])
    return target


def _put_grippers_in_target(
    robot: Articulation,
    target: torch.Tensor,
    left_grip: float,
    right_grip: float,
) -> None:
    left = max(0.0, min(0.041, float(left_grip)))
    right = max(0.0, min(0.041, float(right_grip)))
    right_ids = [robot.joint_names.index(name) for name in RIGHT_GRIPPER_JOINTS]
    left_ids = [robot.joint_names.index(name) for name in LEFT_GRIPPER_JOINTS]
    target[:, right_ids] = right
    target[:, left_ids] = left


def _joint23_to_full_tensor(
    robot: Articulation,
    row: list[float],
    joint23_ids: list[int],
    base_target: torch.Tensor | None = None,
) -> torch.Tensor:
    values = require_vector23(row, "joint23")
    target = robot.data.joint_pos.clone() if base_target is None else base_target.clone()
    target[:, joint23_ids] = torch.tensor(values, dtype=torch.float32, device=robot.device)
    return _clamp_controlled_target(robot, target, joint23_ids)


def _apply_default_head_pose(robot: Articulation) -> torch.Tensor:
    """Initialize the head to the dataset-like downward view used by the global camera."""
    names = ["Head02_Joint", "Head03_Joint"]
    if not all(name in robot.joint_names for name in names):
        missing = [name for name in names if name not in robot.joint_names]
        raise RuntimeError(f"Missing head joint(s): {missing}")
    joint_ids = [robot.joint_names.index(name) for name in names]
    values = torch.tensor([[args_cli.default_head02, args_cli.default_head03]], dtype=torch.float32, device=robot.device)

    joint_pos = robot.data.joint_pos.clone()
    joint_vel = robot.data.joint_vel.clone()
    joint_pos[:, joint_ids] = values
    joint_vel[:, joint_ids] = 0.0
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(values, joint_ids=joint_ids)
    return joint_pos


def _reset_joint23(
    robot: Articulation,
    row: list[float],
    joint23_ids: list[int],
    base_target: torch.Tensor,
) -> torch.Tensor:
    target = _joint23_to_full_tensor(robot, row, joint23_ids, base_target=base_target)
    joint_vel = torch.zeros_like(target)
    robot.write_joint_state_to_sim(target, joint_vel)
    robot.set_joint_position_target(target)
    robot.reset()
    return target


def _compute_ee6d_target(
    sim: SimulationContext,
    robot: Articulation,
    left_ctx: ArmIkContext,
    right_ctx: ArmIkContext,
    row: list[float],
    base_target: torch.Tensor,
    joint23_ids: list[int],
) -> torch.Tensor:
    if len(row) != 20 or not np.all(np.isfinite(np.asarray(row, dtype=np.float64))):
        raise ValueError("ee6d action must contain exactly 20 finite values")
    left_pose_b, left_grip = _ee10_to_pose(row[0:10], sim.device)
    right_pose_b, right_grip = _ee10_to_pose(row[10:20], sim.device)
    left_ctx.controller.set_command(left_pose_b)
    right_ctx.controller.set_command(right_pose_b)
    target = base_target.clone()
    target[:, left_ctx.entity_cfg.joint_ids] = _compute_arm_command(robot, left_ctx)
    target[:, right_ctx.entity_cfg.joint_ids] = _compute_arm_command(robot, right_ctx)
    _put_grippers_in_target(robot, target, left_grip=left_grip, right_grip=right_grip)
    return _clamp_controlled_target(robot, target, joint23_ids)


def _state_snapshot(
    robot: Articulation,
    left_ctx: ArmIkContext,
    right_ctx: ArmIkContext,
    action_target: torch.Tensor,
    joint23_ids: list[int],
    sim_time_s: float,
) -> dict[str, Any]:
    joint_pos = robot.data.joint_pos[0]
    qpos23 = joint_pos[joint23_ids].detach().cpu().tolist()
    action23 = action_target[0, joint23_ids].detach().cpu().tolist()

    root_pose_w = robot.data.root_pose_w
    left_pose_w = robot.data.body_state_w[0, left_ctx.entity_cfg.body_ids[0], 0:7]
    right_pose_w = robot.data.body_state_w[0, right_ctx.entity_cfg.body_ids[0], 0:7]
    left_pos_b, left_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], left_pose_w[None, 0:3], left_pose_w[None, 3:7]
    )
    right_pos_b, right_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], right_pose_w[None, 0:3], right_pose_w[None, 3:7]
    )

    right_grip = 0.0
    left_grip = 0.0
    gripper_joints: dict[str, float] = {}
    if all(name in robot.joint_names for name in RIGHT_GRIPPER_JOINTS):
        right_ids = [robot.joint_names.index(name) for name in RIGHT_GRIPPER_JOINTS]
        right_grip = float(torch.mean(torch.abs(joint_pos[right_ids])).detach().cpu())
        for name, jid in zip(RIGHT_GRIPPER_JOINTS, right_ids):
            gripper_joints[name] = float(joint_pos[jid].detach().cpu())
    if all(name in robot.joint_names for name in LEFT_GRIPPER_JOINTS):
        left_ids = [robot.joint_names.index(name) for name in LEFT_GRIPPER_JOINTS]
        left_grip = float(torch.mean(torch.abs(joint_pos[left_ids])).detach().cpu())
        for name, jid in zip(LEFT_GRIPPER_JOINTS, left_ids):
            gripper_joints[name] = float(joint_pos[jid].detach().cpu())

    left_ee10_base = left_pos_b[0].detach().cpu().tolist() + _quat_to_rot6d(left_quat_b[0]) + [left_grip]
    right_ee10_base = right_pos_b[0].detach().cpu().tolist() + _quat_to_rot6d(right_quat_b[0]) + [right_grip]

    state = {
        "ready": True,
        "mode": COMMAND_BUFFER.command_status()["command_mode"],
        "sim_time_s": float(sim_time_s),
        "joint_names23": list(JOINT_NAMES_23),
        "qpos23": qpos23,
        "action23": action23,
        "action_semantics": "absolute_joint_position_target",
        "ee6d_base": left_ee10_base + right_ee10_base,
        "left_ee_base": left_ee10_base,
        "right_ee_base": right_ee10_base,
        "left_ee_world": left_pose_w.detach().cpu().tolist(),
        "right_ee_world": right_pose_w.detach().cpu().tolist(),
        "gripper_joints": gripper_joints,
        "physical_gripper_indices23": {"left": [19, 20], "right": [10, 11]},
        "gripper_convention": {"open_m": args_cli.gripper_open_m, "close_m": args_cli.gripper_close_m},
        "camera": _camera_buffer().status(),
        "recorder": _recorder_status(),
        "runtime": RUNTIME_METRICS.status(),
    }
    state.update(COMMAND_BUFFER.command_status())
    return state


def _advance_deadline(deadline: float, period: float, current: float) -> float:
    while deadline <= current + 1e-9:
        deadline += period
    return deadline


def _effective_rates(sim_dt: float) -> tuple[float, float, float]:
    if not np.isfinite(sim_dt) or sim_dt <= 0.0:
        raise ValueError(f"--sim_dt must be a positive finite number, got {sim_dt}")
    physics_hz = 1.0 / sim_dt
    render_hz = min(max(float(args_cli.render_hz), 0.1), physics_hz)
    camera_hz = min(max(float(args_cli.camera_hz), 0.1), physics_hz)
    if args_cli.camera_update_every is not None:
        camera_hz = physics_hz / max(1, int(args_cli.camera_update_every))
    return physics_hz, render_hz, camera_hz


def run_simulator(sim: SimulationContext, robot: Articulation, camera_manager: BrxCameraManager) -> None:
    sim_dt = sim.get_physics_dt()
    physics_hz, render_hz, camera_hz = _effective_rates(sim_dt)
    if args_cli.camera_update_every is not None:
        camera_hz = physics_hz / max(1, int(args_cli.camera_update_every))
        print(f"[BRX] --camera_update_every is deprecated; effective camera_hz={camera_hz:.3f}")
    render_period = 1.0 / render_hz
    camera_period = 1.0 / camera_hz

    joint23_ids = resolve_joint_ids(robot.joint_names)
    action_target = _apply_default_head_pose(robot)
    camera_manager.set_initial_poses(sim.device)
    robot.write_data_to_sim()
    sim.step(render=False)
    RUNTIME_METRICS.physics()
    robot.update(sim_dt)

    left_ctx = _resolve_arm(sim, robot, LEFT_ARM_JOINTS, args_cli.left_ee_body)
    right_ctx = _resolve_arm(sim, robot, RIGHT_ARM_JOINTS, args_cli.right_ee_body)
    camera_manager.resolve_robot_bodies(robot)
    camera_manager.update_poses(robot, sim.device)
    for _ in range(max(1, int(args_cli.camera_warmup_renders))):
        sim.render()
        RUNTIME_METRICS.render()
    sim_time_s = sim_dt
    if camera_manager.update_after_render(camera_period, sim_time_s) is not None:
        RUNTIME_METRICS.camera()

    print("[BRX] Control conventions:")
    print("[BRX] ee6d: [left_xyz, left_rot6d, left_gripper, right_xyz, right_rot6d, right_gripper], absolute base frame")
    print("[BRX] joint23/action23: absolute joint targets in AppleVisionPro.joint_contract.JOINT_NAMES_23 order")
    print("[BRX] gripper scalar is interpreted as jaw joint target meters and clamped to [0, 0.041]")
    print(f"[BRX] gripper diagnostic convention: open_m={args_cli.gripper_open_m:.4f}, close_m={args_cli.gripper_close_m:.4f}")
    print(f"[BRX] default head joints: Head02={args_cli.default_head02:.5f}, Head03={args_cli.default_head03:.5f}")
    print(
        f"[BRX] rates: physics={physics_hz:.1f}Hz render={render_hz:.1f}Hz "
        f"camera={camera_hz:.1f}Hz realtime={args_cli.realtime}"
    )
    COMMAND_BUFFER.set_state(
        _state_snapshot(robot, left_ctx, right_ctx, action_target, joint23_ids, sim_time_s)
    )
    openxr_teleop = OpenXRTeleop() if args_cli.enable_openxr_teleop else None

    hold_count = 0
    current_mode: str | None = None
    current_row: list[float] | None = None
    next_render_sim_time = sim_time_s + render_period
    next_camera_sim_time = sim_time_s + camera_period
    next_wall_t = time.monotonic() + sim_dt
    try:
        while simulation_app.is_running():
            state_before = COMMAND_BUFFER.get_state()
            openxr_row = openxr_teleop.advance(state_before) if openxr_teleop is not None and state_before.get("ready") else None
            if openxr_row is not None:
                current_mode = "ee6d"
                current_row = openxr_row
                hold_count = max(1, args_cli.command_hold_steps)
            elif hold_count <= 0:
                current_mode, current_row, _ = COMMAND_BUFFER.next_row()
                hold_count = max(1, args_cli.command_hold_steps)
            hold_count -= 1

            try:
                if current_mode == "joint23" and current_row is not None:
                    action_target = _joint23_to_full_tensor(
                        robot, current_row, joint23_ids, base_target=action_target
                    )
                elif current_mode == "reset_joint23" and current_row is not None:
                    action_target = _reset_joint23(
                        robot, current_row, joint23_ids, base_target=action_target
                    )
                    COMMAND_BUFFER.set_command("stop", [])
                elif current_mode == "ee6d" and current_row is not None:
                    action_target = _compute_ee6d_target(
                        sim,
                        robot,
                        left_ctx,
                        right_ctx,
                        current_row,
                        base_target=action_target,
                        joint23_ids=joint23_ids,
                    )
                elif current_mode == "stop":
                    action_target = robot.data.joint_pos.clone()
                if not torch.all(torch.isfinite(action_target)):
                    raise ValueError("computed joint target contains NaN/Inf")
            except Exception as exc:
                RUNTIME_METRICS.command_error(exc)
                print(f"[BRX][command] rejected {current_mode}: {exc}; holding current joints")
                action_target = robot.data.joint_pos.clone()
                current_mode = "stop"
                current_row = None
                hold_count = 0
                COMMAND_BUFFER.set_command("stop", [])

            robot.set_joint_position_target(action_target)
            robot.write_data_to_sim()
            sim.step(render=False)
            RUNTIME_METRICS.physics()
            robot.update(sim_dt)
            sim_time_s += sim_dt

            camera_due = sim_time_s + 1e-9 >= next_camera_sim_time
            scheduled_render_due = sim_time_s + 1e-9 >= next_render_sim_time
            captured_frame_id: int | None = None
            if camera_due or scheduled_render_due:
                camera_manager.update_poses(robot, sim.device)
                sim.render()
                RUNTIME_METRICS.render()
                if scheduled_render_due:
                    next_render_sim_time = _advance_deadline(
                        next_render_sim_time, render_period, sim_time_s
                    )
                if camera_due:
                    frame_id = camera_manager.update_after_render(camera_period, sim_time_s)
                    next_camera_sim_time = _advance_deadline(
                        next_camera_sim_time, camera_period, sim_time_s
                    )
                    if frame_id is not None:
                        captured_frame_id = frame_id
                        RUNTIME_METRICS.camera()

            state_after = _state_snapshot(
                robot, left_ctx, right_ctx, action_target, joint23_ids, sim_time_s
            )
            COMMAND_BUFFER.set_state(state_after)
            if captured_frame_id is not None and RECORDER is not None:
                camera_snapshot = camera_manager.buffer.snapshot()
                if camera_snapshot is not None:
                    frame_id, frame_sim_time, capture_ns, images = camera_snapshot
                    RECORDER.enqueue(
                        SynchronizedSample(
                            frame_id=frame_id,
                            sim_time_s=frame_sim_time,
                            capture_monotonic_ns=capture_ns,
                            images=images,
                            observation_state=tuple(state_after["qpos23"]),
                            action=tuple(state_after["action23"]),
                        )
                    )

            RUNTIME_METRICS.maybe_report(sim_dt, args_cli.stats_interval_s)
            if args_cli.realtime:
                now = time.monotonic()
                remaining = next_wall_t - now
                if remaining > 0.0:
                    time.sleep(remaining)
                    now = time.monotonic()
                if now - next_wall_t > max(4.0 * sim_dt, 0.25):
                    next_wall_t = now
                    RUNTIME_METRICS.pacing_reset()
                next_wall_t += sim_dt
    finally:
        if openxr_teleop is not None:
            openxr_teleop.close()


def main() -> None:
    global CAMERA_BUFFER, RECORDER

    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.sim_dt, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.0, 1.0, 2.0], [0.0, 0.0, 1.0])
    _spawn_scene()
    camera_manager = BrxCameraManager(
        BrxCameraConfig(
            width=args_cli.camera_width,
            height=args_cli.camera_height,
            pose_mode=args_cli.camera_pose_mode,
            head_offset=tuple(args_cli.head_camera_offset),
            head_forward=tuple(args_cli.head_camera_forward),
            wrist_forward=tuple(args_cli.wrist_camera_forward),
        )
    )
    CAMERA_BUFFER = camera_manager.buffer
    robot = Articulation(cfg=_make_robot_cfg())
    sim.reset()

    _, _, effective_camera_hz = _effective_rates(float(args_cli.sim_dt))
    if args_cli.record_root:
        record_fps = int(args_cli.record_fps or round(effective_camera_hz))
        if record_fps > effective_camera_hz + 1e-6:
            raise ValueError(
                f"record_fps={record_fps} cannot exceed camera_hz={effective_camera_hz:.3f}"
            )
        RECORDER = LeRobotRecorder(
            RecorderConfig(
                root=Path(args_cli.record_root).expanduser().resolve(),
                repo_id=args_cli.record_repo_id,
                fps=record_fps,
                width=args_cli.camera_width,
                height=args_cli.camera_height,
                use_videos=args_cli.record_use_videos,
                queue_size=args_cli.record_queue_size,
                expected_format=args_cli.lerobot_format,
            )
        )
        if args_cli.record_task:
            RECORDER.start_episode(args_cli.record_task)

    server = _start_http_server()
    print("[BRX] Setup complete.")
    try:
        run_simulator(sim, robot, camera_manager)
    finally:
        server.shutdown()
        server.server_close()
        if RECORDER is not None:
            RECORDER.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
