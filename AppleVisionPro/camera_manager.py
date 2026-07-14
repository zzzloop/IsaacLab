"""Four-camera manager for the BRX042501 Vision Pro data path."""

from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg

from AppleVisionPro.joint_contract import CAMERA_LINK_BY_NAME, CAMERA_NAMES


@dataclass(frozen=True)
class BrxCameraConfig:
    width: int = 640
    height: int = 360
    focal_length: float = 12.0
    horizontal_aperture: float = 24.0
    focus_distance: float = 2.0
    clipping_range: tuple[float, float] = (0.05, 20.0)
    pose_mode: str = "link"
    head_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    head_forward: tuple[float, float, float] = (0.25, 0.0, 0.0)
    wrist_forward: tuple[float, float, float] = (0.20, 0.0, -0.12)


class CameraFrameBuffer:
    """Thread-safe synchronized raw RGB frame set with lazy PNG encoding."""

    def __init__(self, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        self._lock = threading.Lock()
        self._images: dict[str, np.ndarray] = {}
        self._png_cache: dict[str, tuple[int, bytes]] = {}
        self._frame_id = 0
        self._sim_time_s = 0.0
        self._capture_monotonic_ns = 0
        self._capture_fps = 0.0
        self._last_capture_t: float | None = None

    def set_frames(self, images: dict[str, np.ndarray], sim_time_s: float) -> int:
        missing = [name for name in CAMERA_NAMES if name not in images]
        if missing:
            raise ValueError(f"Incomplete synchronized camera set: missing {missing}")
        checked: dict[str, np.ndarray] = {}
        for name in CAMERA_NAMES:
            image = np.asarray(images[name])
            if image.shape != (self.height, self.width, 3):
                raise ValueError(
                    f"Camera {name} expected {(self.height, self.width, 3)}, got {image.shape}"
                )
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            checked[name] = np.ascontiguousarray(image).copy()

        now = time.monotonic()
        with self._lock:
            if self._last_capture_t is not None:
                instant = 1.0 / max(now - self._last_capture_t, 1e-9)
                self._capture_fps = instant if self._capture_fps == 0.0 else 0.9 * self._capture_fps + 0.1 * instant
            self._last_capture_t = now
            self._images = checked
            self._frame_id += 1
            self._sim_time_s = float(sim_time_s)
            self._capture_monotonic_ns = time.monotonic_ns()
            self._png_cache.clear()
            return self._frame_id

    def snapshot(self) -> tuple[int, float, int, dict[str, np.ndarray]] | None:
        with self._lock:
            if len(self._images) != len(CAMERA_NAMES):
                return None
            return (
                self._frame_id,
                self._sim_time_s,
                self._capture_monotonic_ns,
                {name: image.copy() for name, image in self._images.items()},
            )

    def get_png(self, name: str) -> bytes | None:
        if name not in CAMERA_NAMES:
            return None
        with self._lock:
            cached = self._png_cache.get(name)
            if cached is not None and cached[0] == self._frame_id:
                return cached[1]
            image = self._images.get(name)
            if image is None:
                return None
            frame_id = self._frame_id
            image_copy = image.copy()

        buffer = io.BytesIO()
        Image.fromarray(image_copy, mode="RGB").save(buffer, format="PNG", compress_level=3)
        png = buffer.getvalue()
        with self._lock:
            if self._frame_id == frame_id:
                self._png_cache[name] = (frame_id, png)
        return png

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "frame_id": self._frame_id,
                "sim_time_s": self._sim_time_s,
                "capture_monotonic_ns": self._capture_monotonic_ns,
                "capture_fps": round(self._capture_fps, 3),
                "available": [name for name in CAMERA_NAMES if name in self._images],
                "width": self.width,
                "height": self.height,
            }


class BrxCameraManager:
    """Own exactly four Isaac Lab RGB sensors in a stable semantic order."""

    _PRIM_NAMES = ("Cam00HeadLeft", "Cam01HeadRight", "Cam02LeftWrist", "Cam03RightWrist")

    def __init__(self, cfg: BrxCameraConfig) -> None:
        if cfg.pose_mode not in ("link", "lookat"):
            raise ValueError(f"Unsupported camera pose mode: {cfg.pose_mode}")
        self.cfg = cfg
        self.buffer = CameraFrameBuffer(width=cfg.width, height=cfg.height)
        self.body_ids: dict[str, int] = {}

        sim_utils.create_prim("/World/Cameras", "Xform")
        for prim_name in self._PRIM_NAMES:
            sim_utils.create_prim(f"/World/Cameras/{prim_name}", "Xform")

        camera_cfg = CameraCfg(
            prim_path="/World/Cameras/Cam.*/CameraSensor",
            update_period=0.0,
            height=cfg.height,
            width=cfg.width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=cfg.focal_length,
                focus_distance=cfg.focus_distance,
                horizontal_aperture=cfg.horizontal_aperture,
                clipping_range=cfg.clipping_range,
            ),
        )
        self.camera = Camera(cfg=camera_cfg)

    def resolve_robot_bodies(self, robot: Articulation) -> None:
        missing = [link for link in CAMERA_LINK_BY_NAME.values() if link not in robot.body_names]
        if missing:
            raise RuntimeError(f"Robot is missing camera marker bodies: {missing}")
        self.body_ids = {
            name: robot.body_names.index(CAMERA_LINK_BY_NAME[name]) for name in CAMERA_NAMES
        }
        for name in CAMERA_NAMES:
            print(
                f"[BRX][camera] {name}: link={CAMERA_LINK_BY_NAME[name]} "
                f"body_index={self.body_ids[name]}"
            )

    def set_initial_poses(self, device: str) -> None:
        # Temporary valid poses used only before the first robot-state update.
        eyes = torch.tensor(
            [[1.35, -1.18, 1.25], [1.35, -1.12, 1.25], [0.72, 0.82, 0.78], [0.72, -0.82, 0.78]],
            dtype=torch.float32,
            device=device,
        )
        targets = torch.tensor(
            [[0.70, -0.03, 0.50], [0.70, 0.03, 0.50], [0.64, 0.10, 0.50], [0.64, -0.10, 0.50]],
            dtype=torch.float32,
            device=device,
        )
        self.camera.set_world_poses_from_view(eyes, targets)

    @staticmethod
    def _quat_apply(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        q_vec = quat[..., 1:4]
        q_w = quat[..., 0:1]
        uv = torch.cross(q_vec, vector, dim=-1)
        uuv = torch.cross(q_vec, uv, dim=-1)
        return vector + 2.0 * (q_w * uv + uuv)

    def update_poses(self, robot: Articulation, device: str) -> None:
        if len(self.body_ids) != len(CAMERA_NAMES):
            raise RuntimeError("Camera robot bodies have not been resolved")
        poses = {
            name: robot.data.body_state_w[0, self.body_ids[name], 0:7] for name in CAMERA_NAMES
        }

        if self.cfg.pose_mode == "link":
            positions = torch.stack([poses[name][0:3] for name in CAMERA_NAMES], dim=0)
            orientations = torch.stack([poses[name][3:7] for name in CAMERA_NAMES], dim=0)
            # BRX camera marker links use +X forward and +Z up, which is Isaac
            # Lab's "world" camera convention.
            self.camera.set_world_poses(positions, orientations, convention="world")
            return

        head_offset = torch.tensor(self.cfg.head_offset, dtype=torch.float32, device=device)
        head_forward = torch.tensor(self.cfg.head_forward, dtype=torch.float32, device=device)
        wrist_forward = torch.tensor(self.cfg.wrist_forward, dtype=torch.float32, device=device)
        eyes: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for name in CAMERA_NAMES:
            pose = poses[name]
            if name.startswith("head_"):
                eye = pose[0:3] + self._quat_apply(pose[3:7], head_offset)
                target = eye + self._quat_apply(pose[3:7], head_forward)
            else:
                eye = pose[0:3]
                target = eye + self._quat_apply(pose[3:7], wrist_forward)
            eyes.append(eye)
            targets.append(target)
        self.camera.set_world_poses_from_view(torch.stack(eyes), torch.stack(targets))

    @staticmethod
    def _rgb_to_numpy(rgb: torch.Tensor) -> np.ndarray:
        array = rgb.detach().cpu().numpy()
        if array.shape[-1] == 4:
            array = array[..., :3]
        if array.dtype != np.uint8:
            if array.size and float(array.max()) <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)

    def update_after_render(self, dt: float, sim_time_s: float) -> int | None:
        self.camera.update(float(dt))
        rgb = self.camera.data.output.get("rgb")
        if rgb is None or rgb.ndim != 4:
            return None
        if int(rgb.shape[0]) != len(CAMERA_NAMES):
            raise RuntimeError(
                f"Expected exactly four RGB camera tensors, got shape {tuple(rgb.shape)}"
            )
        images = {
            name: self._rgb_to_numpy(rgb[index]) for index, name in enumerate(CAMERA_NAMES)
        }
        return self.buffer.set_frames(images, sim_time_s=sim_time_s)

