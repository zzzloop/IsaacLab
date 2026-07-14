"""Asynchronous, synchronized LeRobot recorder for BRX042501.

The module imports LeRobot lazily so simulation and teleoperation can run
without the optional dataset dependency.  The installed LeRobot package owns
the physical v2.1/v3 layout; this recorder supplies one stable feature schema
and uses the version's public create/add_frame/save_episode API.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from AppleVisionPro.joint_contract import (
    CAMERA_FEATURE_BY_NAME,
    CAMERA_NAMES,
    JOINT_NAMES_23,
    require_vector23,
)


@dataclass(frozen=True)
class RecorderConfig:
    root: Path
    repo_id: str
    fps: int
    width: int
    height: int
    robot_type: str = "BRX042501"
    use_videos: bool = True
    image_writer_threads: int = 4
    queue_size: int = 64
    expected_format: Literal["auto", "v2", "v3"] = "auto"


@dataclass(frozen=True)
class SynchronizedSample:
    frame_id: int
    sim_time_s: float
    capture_monotonic_ns: int
    images: dict[str, np.ndarray]
    observation_state: tuple[float, ...]
    action: tuple[float, ...]


@dataclass
class _Control:
    kind: Literal["save", "abort", "finalize", "shutdown"]
    done: threading.Event
    error: BaseException | None = None


class LeRobotRecorder:
    def __init__(self, cfg: RecorderConfig) -> None:
        if cfg.fps <= 0:
            raise ValueError("Recorder fps must be positive")
        self.cfg = cfg
        self.cfg.root.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[SynchronizedSample | _Control] = queue.Queue(maxsize=cfg.queue_size)
        self._lock = threading.Lock()
        self._dataset: Any | None = None
        self._worker_error: BaseException | None = None
        self._active = False
        self._task = ""
        self._episode_frames = 0
        self._total_frames = 0
        self._episodes_saved = 0
        self._dropped_frames = 0
        self._rate_skipped_frames = 0
        self._last_frame_id = -1
        self._last_enqueued_sim_time = float("-inf")
        self._version = "not-loaded"
        self._dataset_format_version = "unknown"
        self._finalized = False
        self._thread = threading.Thread(target=self._worker, name="brx-lerobot-writer", daemon=True)
        self._thread.start()

    @staticmethod
    def _major_version(version: str) -> int | None:
        token = version.split("+", 1)[0].split(".", 1)[0]
        return int(token) if token.isdigit() else None

    def _check_expected_version(self) -> None:
        if self.cfg.expected_format == "auto":
            return
        major = self._major_version(self._dataset_format_version.lstrip("v"))
        expected = 2 if self.cfg.expected_format == "v2" else 3
        if major is not None and major != expected:
            raise RuntimeError(
                f"--lerobot_format={self.cfg.expected_format} requested, but installed "
                f"LeRobot writes dataset format {self._dataset_format_version} "
                f"(package {self._version}). Use the matching LeRobot environment "
                "or select --lerobot_format=auto."
            )

    def _features(self) -> dict[str, dict[str, Any]]:
        image_dtype = "video" if self.cfg.use_videos else "image"
        features: dict[str, dict[str, Any]] = {}
        for camera_name in CAMERA_NAMES:
            features[CAMERA_FEATURE_BY_NAME[camera_name]] = {
                "dtype": image_dtype,
                "shape": (self.cfg.height, self.cfg.width, 3),
                "names": ["height", "width", "channel"],
            }
        motor_names = {"motors": list(JOINT_NAMES_23)}
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES_23),),
            "names": motor_names,
        }
        features["action"] = {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES_23),),
            "names": motor_names,
        }
        return features

    def _ensure_dataset(self) -> Any:
        if self._dataset is not None:
            return self._dataset
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            try:
                from lerobot.datasets import LeRobotDataset
            except ImportError as exc:
                try:
                    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
                except ImportError:
                    raise RuntimeError(
                        "LeRobot is not installed in this Python environment. Install the desired "
                        "LeRobot v3 or v2.1 release before starting recording; teleoperation itself "
                        "does not require LeRobot."
                    ) from exc
        try:
            self._version = importlib.metadata.version("lerobot")
        except importlib.metadata.PackageNotFoundError:
            self._version = "unknown"
        for module_name in (
            "lerobot.datasets.dataset_metadata",
            "lerobot.datasets.lerobot_dataset",
            "lerobot.common.datasets.lerobot_dataset",
        ):
            try:
                module = __import__(module_name, fromlist=["CODEBASE_VERSION"])
                codebase_version = getattr(module, "CODEBASE_VERSION", None)
                if codebase_version is not None:
                    self._dataset_format_version = str(codebase_version)
                    break
            except ImportError:
                continue
        self._check_expected_version()
        create_kwargs = dict(
            repo_id=self.cfg.repo_id,
            root=self.cfg.root,
            fps=self.cfg.fps,
            robot_type=self.cfg.robot_type,
            features=self._features(),
            use_videos=self.cfg.use_videos,
            image_writer_threads=self.cfg.image_writer_threads,
        )
        supported = inspect.signature(LeRobotDataset.create).parameters
        create_kwargs = {key: value for key, value in create_kwargs.items() if key in supported}
        self._dataset = LeRobotDataset.create(**create_kwargs)
        print(
            f"[BRX][record] LeRobot package={self._version} "
            f"format={self._dataset_format_version} dataset created: "
            f"root={self.cfg.root}, repo_id={self.cfg.repo_id}, fps={self.cfg.fps}"
        )
        return self._dataset

    def start_episode(self, task: str) -> dict[str, Any]:
        task = str(task).strip()
        if not task:
            raise ValueError("A non-empty language task is required for every episode")
        # Fail before arming recording if LeRobot is absent, incompatible, or
        # the dataset root cannot be created.
        self._ensure_dataset()
        with self._lock:
            self._raise_if_failed_locked()
            if self._finalized:
                raise RuntimeError("The dataset has been finalized; restart with a new --record_root")
            if self._active:
                raise RuntimeError("An episode is already recording")
            self._active = True
            self._task = task
            self._episode_frames = 0
            self._last_frame_id = -1
            self._last_enqueued_sim_time = float("-inf")
        return self.status()

    def enqueue(self, sample: SynchronizedSample) -> bool:
        with self._lock:
            self._raise_if_failed_locked()
            if not self._active:
                return False
            if sample.frame_id <= self._last_frame_id:
                return False
            min_period = 1.0 / self.cfg.fps
            if sample.sim_time_s + 1e-9 < self._last_enqueued_sim_time + min_period:
                self._rate_skipped_frames += 1
                return False
            self._last_frame_id = sample.frame_id
            self._last_enqueued_sim_time = sample.sim_time_s
        try:
            self._queue.put_nowait(sample)
            return True
        except queue.Full:
            with self._lock:
                self._dropped_frames += 1
            return False

    def stop_episode(self, save: bool = True, timeout_s: float = 120.0) -> dict[str, Any]:
        with self._lock:
            self._raise_if_failed_locked()
            if not self._active:
                raise RuntimeError("No episode is recording")
            self._active = False
        self._send_control("save" if save else "abort", timeout_s=timeout_s)
        return self.status()

    def finalize(self, timeout_s: float = 120.0) -> None:
        with self._lock:
            active = self._active
            finalized = self._finalized
        if finalized:
            return
        if active:
            self.stop_episode(save=True, timeout_s=timeout_s)
        self._send_control("finalize", timeout_s=timeout_s)
        with self._lock:
            self._finalized = True

    def close(self, timeout_s: float = 120.0) -> None:
        try:
            self.finalize(timeout_s=timeout_s)
        finally:
            self._send_control("shutdown", timeout_s=timeout_s)
            self._thread.join(timeout=max(timeout_s, 1.0))

    def _send_control(self, kind: Literal["save", "abort", "finalize", "shutdown"], timeout_s: float) -> None:
        command = _Control(kind=kind, done=threading.Event())
        self._queue.put(command, timeout=timeout_s)
        if not command.done.wait(timeout=timeout_s):
            raise TimeoutError(f"Timed out waiting for recorder command: {kind}")
        if command.error is not None:
            raise RuntimeError(f"Recorder command {kind} failed: {command.error}") from command.error

    def _validate_sample(self, sample: SynchronizedSample) -> dict[str, Any]:
        state = np.asarray(require_vector23(sample.observation_state, "observation.state"), dtype=np.float32)
        action = np.asarray(require_vector23(sample.action, "action"), dtype=np.float32)
        frame: dict[str, Any] = {
            "observation.state": state,
            "action": action,
            "task": self._task,
        }
        for camera_name in CAMERA_NAMES:
            if camera_name not in sample.images:
                raise ValueError(f"Synchronized sample is missing {camera_name}")
            image = np.asarray(sample.images[camera_name])
            expected_shape = (self.cfg.height, self.cfg.width, 3)
            if image.shape != expected_shape or image.dtype != np.uint8:
                raise ValueError(
                    f"{camera_name} must be uint8 {expected_shape}, got {image.dtype} {image.shape}"
                )
            frame[CAMERA_FEATURE_BY_NAME[camera_name]] = np.ascontiguousarray(image)
        return frame

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if isinstance(item, SynchronizedSample):
                    dataset = self._ensure_dataset()
                    dataset.add_frame(self._validate_sample(item))
                    with self._lock:
                        self._episode_frames += 1
                        self._total_frames += 1
                else:
                    try:
                        if item.kind == "save":
                            dataset = self._ensure_dataset()
                            if self._episode_frames <= 0:
                                raise RuntimeError("Refusing to save an empty episode")
                            dataset.save_episode()
                            with self._lock:
                                self._episodes_saved += 1
                        elif item.kind == "abort":
                            if self._dataset is not None:
                                clear = getattr(self._dataset, "clear_episode_buffer", None)
                                if clear is None:
                                    raise RuntimeError("Installed LeRobot cannot abort an episode safely")
                                clear()
                        elif item.kind == "finalize":
                            if self._dataset is not None:
                                finalize = getattr(self._dataset, "finalize", None)
                                consolidate = getattr(self._dataset, "consolidate", None)
                                if callable(finalize):
                                    finalize()
                                elif callable(consolidate):
                                    # LeRobot v2.1 used consolidate() before v3 introduced finalize().
                                    consolidate()
                        elif item.kind == "shutdown":
                            return
                    except BaseException as exc:
                        item.error = exc
                    finally:
                        item.done.set()
            except BaseException as exc:
                with self._lock:
                    self._worker_error = exc
                    self._active = False
                if isinstance(item, _Control):
                    item.error = exc
                    item.done.set()
            finally:
                self._queue.task_done()

    def _raise_if_failed_locked(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError(f"LeRobot writer failed: {self._worker_error}") from self._worker_error

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": True,
                "active": self._active,
                "task": self._task,
                "episode_frames": self._episode_frames,
                "total_frames": self._total_frames,
                "episodes_saved": self._episodes_saved,
                "dropped_frames": self._dropped_frames,
                "rate_skipped_frames": self._rate_skipped_frames,
                "queue_depth": self._queue.qsize(),
                "queue_capacity": self.cfg.queue_size,
                "root": str(self.cfg.root),
                "repo_id": self.cfg.repo_id,
                "fps": self.cfg.fps,
                "lerobot_version": self._version,
                "dataset_format_version": self._dataset_format_version,
                "finalized": self._finalized,
                "error": None if self._worker_error is None else str(self._worker_error),
            }


def disabled_recorder_status() -> dict[str, Any]:
    return {
        "enabled": False,
        "active": False,
        "reason": "Start the server with --record_root to enable LeRobot recording",
    }
