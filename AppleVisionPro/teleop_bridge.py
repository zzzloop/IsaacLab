#!/usr/bin/env python3
"""Apple Vision Pro HTTP pose bridge for BRX042501.

Input quaternions use visionOS/ARKit's common ``[x, y, z, w]`` serialization.
The bridge calibrates hand poses against the current robot end-effector poses,
maps metric deltas 1:1 by default, and sends absolute 20-D EE commands to the
Isaac Lab server.  The server's small DLS adapter converts those poses to the
23-D absolute joint-position targets that are also recorded as dataset action.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge Vision Pro hand poses to BRX042501.")
    parser.add_argument("--listen_host", default="0.0.0.0")
    parser.add_argument("--listen_port", type=int, default=8899)
    parser.add_argument("--brx_url", default="http://127.0.0.1:8765")
    parser.add_argument("--rate_hz", type=float, default=30.0)
    parser.add_argument("--scale", type=float, default=1.0, help="Metric hand-delta to robot-delta scale.")
    parser.add_argument("--axis_map", default="x,y,z", help="Vision axes mapped to robot base axes, e.g. 'z,x,y'.")
    parser.add_argument("--position_only", action="store_true", help="Hold current EE orientations during initial bring-up.")
    parser.add_argument("--max_step_m", type=float, default=0.04)
    parser.add_argument("--max_rotation_step_deg", type=float, default=12.0)
    parser.add_argument("--min_x", type=float, default=-1.20)
    parser.add_argument("--max_x", type=float, default=1.20)
    parser.add_argument("--min_y", type=float, default=-1.20)
    parser.add_argument("--max_y", type=float, default=1.20)
    parser.add_argument("--min_z", type=float, default=0.35)
    parser.add_argument("--max_z", type=float, default=1.35)
    parser.add_argument("--pinch_close_threshold", type=float, default=0.75)
    parser.add_argument("--gripper_open_m", type=float, default=0.0)
    parser.add_argument("--gripper_closed_m", type=float, default=0.041)
    parser.add_argument("--tracking_timeout_s", type=float, default=0.35)
    parser.add_argument("--request_timeout_s", type=float, default=2.0)
    parser.add_argument("--default_task", default="Teleoperate BRX042501 to complete the demonstrated task.")
    return parser


ARGS = _build_parser().parse_args()


def _url(path: str) -> str:
    return ARGS.brx_url.rstrip("/") + path


def _request(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        _url(path),
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=ARGS.request_timeout_s) as response:
            data = response.read()
            return {} if not data else json.loads(data.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"BRX HTTP {exc.code} {path}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach BRX server at {ARGS.brx_url}: {exc}") from exc


def _axis_matrix(spec: str) -> np.ndarray:
    axes = {"x": 0, "y": 1, "z": 2}
    matrix = np.zeros((3, 3), dtype=np.float64)
    tokens = [token.strip().lower() for token in spec.split(",")]
    if len(tokens) != 3:
        raise ValueError("--axis_map must contain exactly three comma-separated signed axes")
    used: set[int] = set()
    for output_axis, token in enumerate(tokens):
        sign = -1.0 if token.startswith("-") else 1.0
        token = token[1:] if token.startswith("-") else token
        if token not in axes or axes[token] in used:
            raise ValueError(f"Invalid --axis_map: {spec!r}")
        used.add(axes[token])
        matrix[output_axis, axes[token]] = sign
    determinant = float(np.linalg.det(matrix))
    if determinant < 0.999:
        raise ValueError(
            f"--axis_map must be a right-handed rotation (det=+1), got det={determinant:.1f}: {spec!r}"
        )
    return matrix


AXIS_MATRIX = _axis_matrix(ARGS.axis_map)


def _finite_vector(value: Any, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be {size} finite numbers, got {value!r}")
    return result


def _quat_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion / max(float(np.linalg.norm(quaternion)), 1e-12)
    x, y, z, w = quaternion
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    first = rot6d[0:3]
    second = rot6d[3:6]
    first = first / max(float(np.linalg.norm(first)), 1e-12)
    second = second - np.dot(first, second) * first
    second = second / max(float(np.linalg.norm(second)), 1e-12)
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=1)


def _matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([matrix[:, 0], matrix[:, 1]])


def _axis_angle(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-8:
        return np.asarray([1.0, 0.0, 0.0]), 0.0
    if math.pi - angle < 1e-5:
        values, vectors = np.linalg.eigh((matrix + np.eye(3)) * 0.5)
        axis = vectors[:, int(np.argmax(values))]
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        return axis, angle
    axis = np.asarray(
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]],
        dtype=np.float64,
    ) / (2.0 * math.sin(angle))
    return axis, angle


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis / max(float(np.linalg.norm(axis)), 1e-12)
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _clip_rotation(target: np.ndarray, previous: np.ndarray) -> np.ndarray:
    delta = target @ previous.T
    axis, angle = _axis_angle(delta)
    max_angle = math.radians(max(ARGS.max_rotation_step_deg, 0.0))
    if angle <= max_angle or max_angle <= 0.0:
        return target if max_angle > 0.0 else previous
    return _rotation(axis, max_angle) @ previous


def _clip_position(target: np.ndarray, previous: np.ndarray) -> np.ndarray:
    target = target.copy()
    target[0] = np.clip(target[0], ARGS.min_x, ARGS.max_x)
    target[1] = np.clip(target[1], ARGS.min_y, ARGS.max_y)
    target[2] = np.clip(target[2], ARGS.min_z, ARGS.max_z)
    delta = target - previous
    distance = float(np.linalg.norm(delta))
    if distance > ARGS.max_step_m:
        target = previous + delta / max(distance, 1e-12) * ARGS.max_step_m
    return target


def _gripper_target(normalized_pinch: float) -> float:
    closed_ratio = float(np.clip(normalized_pinch / max(ARGS.pinch_close_threshold, 1e-6), 0.0, 1.0))
    return ARGS.gripper_open_m + closed_ratio * (ARGS.gripper_closed_m - ARGS.gripper_open_m)


@dataclass(frozen=True)
class HandPose:
    position: np.ndarray
    rotation: np.ndarray
    pinch: float


@dataclass(frozen=True)
class Calibration:
    left_hand: HandPose
    right_hand: HandPose
    left_robot_position: np.ndarray
    right_robot_position: np.ndarray
    left_robot_rotation: np.ndarray
    right_robot_rotation: np.ndarray


def _parse_hand(payload: dict[str, Any], side: str) -> HandPose:
    hand = payload.get(side)
    if not isinstance(hand, dict):
        raise ValueError(f"Missing {side} hand object")
    if hand.get("tracking", True) is False:
        raise ValueError(f"{side} hand tracking is invalid")
    position = _finite_vector(hand.get("position"), 3, f"{side}.position")
    quaternion = _finite_vector(hand.get("quaternion"), 4, f"{side}.quaternion")
    if float(np.linalg.norm(quaternion)) < 1e-8:
        raise ValueError(f"{side}.quaternion has zero length")
    pinch = float(hand.get("pinch", 0.0))
    if not math.isfinite(pinch):
        raise ValueError(f"{side}.pinch must be finite")
    return HandPose(position=position, rotation=_quat_xyzw_to_matrix(quaternion), pinch=pinch)


class VisionProBridge:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.calibration: Calibration | None = None
        self.last_send_t = 0.0
        self.last_packet_t = 0.0
        self.last_sequence: int | None = None
        self.accepted_packets = 0
        self.rate_limited_packets = 0
        self.record_requested = False
        self.watchdog_stop_sent = False

    def _calibrate(self, left: HandPose, right: HandPose, state: dict[str, Any]) -> None:
        ee = _finite_vector(state["ee6d_base"], 20, "BRX ee6d_base")
        self.calibration = Calibration(
            left_hand=left,
            right_hand=right,
            left_robot_position=ee[0:3].copy(),
            right_robot_position=ee[10:13].copy(),
            left_robot_rotation=_rot6d_to_matrix(ee[3:9]),
            right_robot_rotation=_rot6d_to_matrix(ee[13:19]),
        )
        print(
            "[visionpro] calibrated "
            f"left={left.position.round(4).tolist()} right={right.position.round(4).tolist()}"
        )

    @staticmethod
    def _mapped_rotation(rotation: np.ndarray) -> np.ndarray:
        return AXIS_MATRIX @ rotation @ AXIS_MATRIX.T

    def _target_rotation(
        self,
        current_hand: HandPose,
        calibration_hand: HandPose,
        robot_at_calibration: np.ndarray,
        current_robot: np.ndarray,
    ) -> np.ndarray:
        current_mapped = self._mapped_rotation(current_hand.rotation)
        calibration_mapped = self._mapped_rotation(calibration_hand.rotation)
        world_delta = current_mapped @ calibration_mapped.T
        return _clip_rotation(world_delta @ robot_at_calibration, current_robot)

    def _handle_record_edge(self, payload: dict[str, Any]) -> None:
        if "record" not in payload:
            return
        requested = bool(payload["record"])
        if requested and not self.record_requested:
            task = str(payload.get("task", ARGS.default_task)).strip()
            _request("POST", "/record/start", {"task": task})
        elif not requested and self.record_requested:
            _request("POST", "/record/stop", {})
        self.record_requested = requested

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = time.monotonic()
        if payload.get("clutch", payload.get("enabled", True)) is False:
            self.last_packet_t = now
            self.watchdog_stop_sent = False
            _request("POST", "/command/stop", {})
            return {"ok": True, "holding": True, "reason": "clutch_released"}

        sequence = payload.get("sequence")
        if sequence is not None:
            sequence = int(sequence)
            if self.last_sequence is not None and sequence <= self.last_sequence:
                return {"ok": True, "skipped": "stale_sequence", "sequence": sequence}
            self.last_sequence = sequence

        left = _parse_hand(payload, "left")
        right = _parse_hand(payload, "right")
        # Only a packet with two valid tracked hands keeps the motion watchdog
        # alive.  Malformed or tracking=false packets must not mask a dropout.
        self.last_packet_t = now
        self.watchdog_stop_sent = False
        min_period = 1.0 / max(ARGS.rate_hz, 1e-6)
        if now - self.last_send_t < min_period:
            self.rate_limited_packets += 1
            return {"ok": True, "skipped": "rate_limit"}

        state = _request("GET", "/state")
        if not state.get("ready"):
            raise RuntimeError("BRX simulation is not ready")
        ee = _finite_vector(state["ee6d_base"], 20, "BRX ee6d_base")
        if self.calibration is None or bool(payload.get("calibrate", False)):
            self._calibrate(left, right, state)
        assert self.calibration is not None

        action = ee.copy()
        action[0:3] = _clip_position(
            self.calibration.left_robot_position
            + ARGS.scale * (AXIS_MATRIX @ (left.position - self.calibration.left_hand.position)),
            ee[0:3],
        )
        action[10:13] = _clip_position(
            self.calibration.right_robot_position
            + ARGS.scale * (AXIS_MATRIX @ (right.position - self.calibration.right_hand.position)),
            ee[10:13],
        )
        if not ARGS.position_only:
            action[3:9] = _matrix_to_rot6d(
                self._target_rotation(
                    left,
                    self.calibration.left_hand,
                    self.calibration.left_robot_rotation,
                    _rot6d_to_matrix(ee[3:9]),
                )
            )
            action[13:19] = _matrix_to_rot6d(
                self._target_rotation(
                    right,
                    self.calibration.right_hand,
                    self.calibration.right_robot_rotation,
                    _rot6d_to_matrix(ee[13:19]),
                )
            )
        action[9] = _gripper_target(left.pinch)
        action[19] = _gripper_target(right.pinch)

        reply = _request("POST", "/command/ee6d", {"action": action.astype(float).tolist()})
        self._handle_record_edge(payload)
        self.last_send_t = now
        self.accepted_packets += 1
        return {
            "ok": True,
            "brx": reply,
            "sequence": sequence,
            "calibrated": True,
            "orientation_enabled": not ARGS.position_only,
            "left_xyz": action[0:3].round(5).tolist(),
            "right_xyz": action[10:13].round(5).tolist(),
            "left_gripper_m": round(float(action[9]), 5),
            "right_gripper_m": round(float(action[19]), 5),
        }

    def status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "calibrated": self.calibration is not None,
            "accepted_packets": self.accepted_packets,
            "rate_limited_packets": self.rate_limited_packets,
            "last_sequence": self.last_sequence,
            "last_packet_age_s": None if self.last_packet_t == 0.0 else round(time.monotonic() - self.last_packet_t, 4),
            "record_requested": self.record_requested,
            "axis_map": ARGS.axis_map,
            "scale": ARGS.scale,
            "position_only": ARGS.position_only,
        }

    def watchdog(self) -> None:
        while True:
            time.sleep(min(max(ARGS.tracking_timeout_s / 4.0, 0.02), 0.1))
            with self.lock:
                if self.last_packet_t == 0.0 or self.watchdog_stop_sent:
                    continue
                if time.monotonic() - self.last_packet_t <= ARGS.tracking_timeout_s:
                    continue
                try:
                    _request("POST", "/command/stop", {})
                    self.watchdog_stop_sent = True
                    print("[visionpro] tracking timeout: BRX is holding current joint positions")
                    if self.record_requested:
                        _request("POST", "/record/stop", {})
                        self.record_requested = False
                        print("[visionpro] tracking timeout: active LeRobot episode saved and stopped")
                except Exception as exc:
                    print(f"[visionpro] watchdog stop failed: {exc}")


BRIDGE = VisionProBridge()


class Handler(BaseHTTPRequestHandler):
    server_version = "BRXVisionProTeleop/0.2"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/health", "/status"):
            with BRIDGE.lock:
                self._json(200, BRIDGE.status())
        else:
            self._json(404, {"ok": False, "error": "unknown endpoint"})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            with BRIDGE.lock:
                if self.path == "/teleop":
                    response = BRIDGE.handle(payload)
                elif self.path == "/calibrate":
                    payload["calibrate"] = True
                    response = BRIDGE.handle(payload)
                elif self.path == "/record/start":
                    response = _request("POST", "/record/start", payload)
                    BRIDGE.record_requested = True
                elif self.path == "/record/stop":
                    response = _request("POST", "/record/stop", payload)
                    BRIDGE.record_requested = False
                elif self.path == "/record/abort":
                    response = _request("POST", "/record/abort", payload)
                    BRIDGE.record_requested = False
                else:
                    self._json(404, {"ok": False, "error": "unknown endpoint"})
                    return
            self._json(200, response)
        except Exception as exc:
            self._json(400, {"ok": False, "error": str(exc)})


def main() -> None:
    print(f"[visionpro] listening on http://{ARGS.listen_host}:{ARGS.listen_port}")
    print(
        f"[visionpro] BRX={ARGS.brx_url} rate={ARGS.rate_hz:.1f}Hz "
        f"axis_map={ARGS.axis_map} scale={ARGS.scale} position_only={ARGS.position_only}"
    )
    threading.Thread(target=BRIDGE.watchdog, name="visionpro-watchdog", daemon=True).start()
    server = ThreadingHTTPServer((ARGS.listen_host, ARGS.listen_port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
