# SPDX-License-Identifier: BSD-3-Clause

"""Vision Pro teleoperation bridge for BRX data collection.

This script receives simple HTTP JSON packets from a visionOS app and forwards
them to ``brx_control_server.py`` as ee6d commands.  It can also record an
ACT-style HDF5 episode:

  /action                         [T, 23] float32
  /observations/qpos              [T, 23] float32
  /observations/images/left_eye   [T, H, W, 3] uint8
  /observations/images/left_wrist [T, H, W, 3] uint8
  /observations/images/right_wrist[T, H, W, 3] uint8

Vision Pro packet format for POST /teleop:

{
  "calibrate": false,
  "record": true,
  "left":  {"position": [x, y, z], "quaternion": [x, y, z, w], "pinch": 0.0},
  "right": {"position": [x, y, z], "quaternion": [x, y, z, w], "pinch": 0.0}
}

The bridge maps hand position deltas from the calibration frame to robot-base
EE target deltas:

  target_xyz = brx_ee_xyz_at_calibration + scale * axis_map(hand_xyz - hand_xyz_at_calibration)

By default orientation is kept from the current robot EE pose; this makes the
first version robust while you tune position and gripper mapping.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image


parser = argparse.ArgumentParser(description="Receive Vision Pro teleop packets and drive BRX /command/ee6d.")
parser.add_argument("--listen_host", type=str, default="0.0.0.0")
parser.add_argument("--listen_port", type=int, default=8899)
parser.add_argument("--brx_url", type=str, default="http://127.0.0.1:8765")
parser.add_argument("--rate_hz", type=float, default=30.0, help="Maximum command/record rate.")
parser.add_argument("--scale", type=float, default=1.0, help="Hand-motion to robot-motion scale.")
parser.add_argument(
    "--axis_map",
    type=str,
    default="x,y,z",
    help="Map Vision Pro delta axes into robot base axes, e.g. 'z,-x,y'.",
)
parser.add_argument("--max_step_m", type=float, default=0.04, help="Max EE target movement per received packet.")
parser.add_argument("--min_z", type=float, default=0.35)
parser.add_argument("--max_z", type=float, default=1.35)
parser.add_argument("--gripper_max", type=float, default=0.041)
parser.add_argument("--pinch_close_threshold", type=float, default=0.75, help="pinch>=threshold means closed.")
parser.add_argument("--record_path", type=str, default=None, help="Output HDF5 path. If omitted, no HDF5 is written.")
parser.add_argument("--record_always", action="store_true", help="Record every accepted packet regardless of payload.record.")
parser.add_argument("--image_height", type=int, default=360)
parser.add_argument("--image_width", type=int, default=640)
parser.add_argument("--timeout", type=float, default=2.0)
args = parser.parse_args()


def _join_url(base: str, path: str) -> str:
    return base.rstrip("/") + path


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=args.timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_png(url: str) -> np.ndarray:
    with urllib.request.urlopen(url, timeout=args.timeout) as resp:
        image = Image.open(BytesIO(resp.read())).convert("RGB")
    if image.size != (args.image_width, args.image_height):
        image = image.resize((args.image_width, args.image_height), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _parse_axis_map(spec: str) -> list[tuple[int, float]]:
    axes = {"x": 0, "y": 1, "z": 2}
    items = []
    for raw in spec.split(","):
        token = raw.strip().lower()
        sign = -1.0 if token.startswith("-") else 1.0
        token = token[1:] if token.startswith("-") else token
        if token not in axes:
            raise ValueError(f"Bad axis_map token: {raw!r}")
        items.append((axes[token], sign))
    if len(items) != 3:
        raise ValueError("--axis_map must contain exactly 3 axes")
    return items


AXIS_MAP = _parse_axis_map(args.axis_map)


def _map_delta(delta: np.ndarray) -> np.ndarray:
    return np.asarray([sign * delta[idx] for idx, sign in AXIS_MAP], dtype=np.float32)


def _clip_xyz(target: np.ndarray, previous: np.ndarray) -> np.ndarray:
    out = target.astype(np.float32).copy()
    out[2] = np.clip(out[2], args.min_z, args.max_z)
    delta = out - previous
    dist = float(np.linalg.norm(delta))
    if dist > args.max_step_m:
        out = previous + delta / max(dist, 1e-8) * args.max_step_m
    return out


def _gripper_from_pinch(pinch: float) -> float:
    # Vision pinch high means fingers closed. BRX scalar is jaw opening meters.
    t = np.clip(float(pinch) / max(args.pinch_close_threshold, 1e-6), 0.0, 1.0)
    return float((1.0 - t) * args.gripper_max)


@dataclass
class Calibration:
    left_hand: np.ndarray
    right_hand: np.ndarray
    left_ee: np.ndarray
    right_ee: np.ndarray


class Hdf5Recorder:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.qpos: list[np.ndarray] = []
        self.action: list[np.ndarray] = []
        self.images: dict[str, list[np.ndarray]] = {"left_eye": [], "left_wrist": [], "right_wrist": []}

    def append(self, qpos: np.ndarray, action: np.ndarray) -> None:
        self.qpos.append(qpos.astype(np.float32))
        self.action.append(action.astype(np.float32))
        for name in self.images:
            try:
                frame = _get_png(_join_url(args.brx_url, f"/camera/{name}.png"))
            except Exception:
                frame = np.zeros((args.image_height, args.image_width, 3), dtype=np.uint8)
            self.images[name].append(frame)

    def save(self) -> None:
        if not self.qpos:
            return
        with h5py.File(self.path, "w") as f:
            f.create_dataset("action", data=np.stack(self.action, axis=0), dtype="float32")
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.stack(self.qpos, axis=0), dtype="float32")
            img_group = obs.create_group("images")
            for name, frames in self.images.items():
                img_group.create_dataset(name, data=np.stack(frames, axis=0), dtype="uint8")


class Bridge:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.calib: Calibration | None = None
        self.last_send_t = 0.0
        self.recorder = Hdf5Recorder(args.record_path) if args.record_path else None

    def calibrate(self, payload: dict[str, Any], state: dict[str, Any]) -> None:
        left_hand = np.asarray(payload["left"]["position"], dtype=np.float32)
        right_hand = np.asarray(payload["right"]["position"], dtype=np.float32)
        ee6d = np.asarray(state["ee6d_base"], dtype=np.float32)
        self.calib = Calibration(left_hand=left_hand, right_hand=right_hand, left_ee=ee6d[0:3].copy(), right_ee=ee6d[10:13].copy())
        print("[visionpro] calibrated:", {"left_hand": left_hand.round(4).tolist(), "right_hand": right_hand.round(4).tolist()})

    def handle_packet(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = time.monotonic()
        min_dt = 1.0 / max(args.rate_hz, 1e-6)
        if now - self.last_send_t < min_dt:
            return {"ok": True, "skipped": "rate_limit"}

        state = _get_json(_join_url(args.brx_url, "/state"))
        if not state.get("ready"):
            raise RuntimeError(f"BRX server not ready: {state}")
        if self.calib is None or payload.get("calibrate"):
            self.calibrate(payload, state)

        assert self.calib is not None
        ee6d = np.asarray(state["ee6d_base"], dtype=np.float32)
        action = ee6d.copy()

        left_hand = np.asarray(payload["left"]["position"], dtype=np.float32)
        right_hand = np.asarray(payload["right"]["position"], dtype=np.float32)
        left_xyz = self.calib.left_ee + args.scale * _map_delta(left_hand - self.calib.left_hand)
        right_xyz = self.calib.right_ee + args.scale * _map_delta(right_hand - self.calib.right_hand)
        action[0:3] = _clip_xyz(left_xyz, ee6d[0:3])
        action[10:13] = _clip_xyz(right_xyz, ee6d[10:13])
        action[9] = _gripper_from_pinch(float(payload.get("left", {}).get("pinch", 0.0)))
        action[19] = _gripper_from_pinch(float(payload.get("right", {}).get("pinch", 0.0)))

        reply = _post_json(_join_url(args.brx_url, "/command/ee6d"), {"action": action.astype(float).tolist()})
        self.last_send_t = now

        if self.recorder is not None and (args.record_always or payload.get("record", False)):
            qpos = np.asarray(state["qpos23"], dtype=np.float32)
            # For ee6d teleop, use the observed joint target proxy. This matches
            # absolute-qpos ACT layout and is adequate when the low-level servo tracks closely.
            self.recorder.append(qpos=qpos, action=qpos)

        return {
            "ok": True,
            "brx": reply,
            "left_xyz": action[0:3].round(4).tolist(),
            "right_xyz": action[10:13].round(4).tolist(),
            "left_grip": round(float(action[9]), 5),
            "right_grip": round(float(action[19]), 5),
            "recorded": 0 if self.recorder is None else len(self.recorder.qpos),
        }

    def close(self) -> None:
        if self.recorder is not None:
            self.recorder.save()
            print(f"[visionpro] saved {len(self.recorder.qpos)} frames to {self.recorder.path}")


BRIDGE = Bridge()


class Handler(BaseHTTPRequestHandler):
    server_version = "BRXVisionProTeleop/0.1"

    def log_message(self, fmt: str, *args_: Any) -> None:
        return

    def _send_json(self, code: int, data: dict[str, Any]) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"ok": True, "calibrated": BRIDGE.calib is not None})
        else:
            self._send_json(404, {"ok": False, "error": "unknown endpoint"})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path == "/teleop":
                with BRIDGE.lock:
                    self._send_json(200, BRIDGE.handle_packet(payload))
            elif self.path == "/save":
                BRIDGE.close()
                self._send_json(200, {"ok": True})
            else:
                self._send_json(404, {"ok": False, "error": "unknown endpoint"})
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})


def main() -> None:
    print(f"[visionpro] listening on http://{args.listen_host}:{args.listen_port}")
    print("[visionpro] POST /teleop with left/right position, quaternion, pinch")
    if args.record_path:
        print(f"[visionpro] recording to {args.record_path}")
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        BRIDGE.close()


if __name__ == "__main__":
    main()
