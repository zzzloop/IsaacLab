# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""
BRX Isaac Lab control server.

Starts the BRX URDF robot in the table-top task scene and exposes two local HTTP
control interfaces:

- POST /command/ee6d
  Body: {"action": [20]} or {"action": [[20], ...]}
  Convention: [left_xyz(3), left_rot6d(6), left_gripper(1), right_xyz(3), right_rot6d(6), right_gripper(1)]
  The EE pose is absolute in the robot base frame, matching the X-VLA custom_handler.py FK adapter.

- POST /command/joint23
  Body: {"qpos": [23]} or {"qpos": [[23], ...]}
  Convention: absolute joint targets in the 23D FK/training order listed in EXPECTED_MOVABLE_JOINTS.

- GET /state
  Returns current qpos in the 23D FK/training order plus left/right EE world poses.

Example:
    ./isaaclab.sh -p scripts/custom/brx_control_server.py \
        --urdf_path /home/kemove/zzk_data/IsaacLab/BRX042501_wheel.urdf \
        --force_usd_conversion --no_instanceable
"""

from __future__ import annotations

import argparse
import json
import os
import io
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="BRX Isaac Lab HTTP control server.")
parser.add_argument("--urdf_path", type=str, default="/home/kemove/zzk_data/IsaacLab/BRX042501_wheel.urdf")
parser.add_argument("--usd_dir", type=str, default=None)
parser.add_argument("--force_usd_conversion", action="store_true")
parser.add_argument("--no_instanceable", action="store_true")
parser.add_argument("--robot_prim", type=str, default="/World/Robot")
parser.add_argument("--left_ee_body", type=str, default="LinearclampinggripperJZ02_Link")
parser.add_argument("--right_ee_body", type=str, default="LinearclampinggripperJZ01_Link")
parser.add_argument("--head_camera_body", type=str, default="EyeL_Link", help="Robot body used by /camera/head.png.")
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--command_hold_steps", type=int, default=1, help="Simulation steps to hold each row of a command chunk.")
parser.add_argument("--joint_stiffness", type=float, default=2500.0, help="Position drive stiffness for all imported robot joints.")
parser.add_argument("--joint_damping", type=float, default=120.0, help="Position drive damping for all imported robot joints.")
parser.add_argument("--effort_limit", type=float, default=800.0, help="Implicit actuator effort limit.")
parser.add_argument("--velocity_limit", type=float, default=60.0, help="Implicit actuator velocity limit.")
parser.add_argument("--no_task_scene", action="store_true")
parser.add_argument("--camera_width", type=int, default=640)
parser.add_argument("--camera_height", type=int, default=480)
parser.add_argument(
    "--left_wrist_camera_offset",
    type=float,
    nargs=3,
    default=(0.04, 0.0, 0.04),
    help="Left wrist camera local xyz offset in the left EE body frame.",
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
    "--right_wrist_camera_offset",
    type=float,
    nargs=3,
    default=(0.04, 0.0, 0.04),
    help="Right wrist camera local xyz offset in the right EE body frame.",
)
parser.add_argument(
    "--wrist_camera_forward",
    type=float,
    nargs=3,
    default=(0.25, 0.0, 0.0),
    help="Local look-at vector for wrist cameras in each EE body frame.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import torch.nn.functional as F
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix, subtract_frame_transforms


EXPECTED_MOVABLE_JOINTS = [
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
    "JawBlock01_Joint",
    "JawBlock02_Joint",
    "ArmR02_Joint",
    "ArmR03_Joint",
    "ArmR04_Joint",
    "ArmR05_Joint",
    "ArmR06_Joint",
    "ArmR07_Joint",
    "ArmR08_Joint",
    "JawBlock03_Joint",
    "JawBlock04_Joint",
    "Head02_Joint",
    "Head03_Joint",
]
LEFT_ARM_JOINTS = [f"ArmL0{i}_Joint" for i in range(2, 9)]
RIGHT_ARM_JOINTS = [f"ArmR0{i}_Joint" for i in range(2, 9)]
LEFT_GRIPPER_JOINTS = ["JawBlock03_Joint", "JawBlock04_Joint"]
RIGHT_GRIPPER_JOINTS = ["JawBlock01_Joint", "JawBlock02_Joint"]
CAMERA_NAMES = ["head", "left_wrist", "right_wrist"]


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
        self.state: dict[str, Any] = {"ready": False}

    def set_command(self, mode: str, rows: list[list[float]]) -> int:
        with self._lock:
            self.mode = mode
            self.queue = [list(row) for row in rows]
            self.last = None
            self.version += 1
            return self.version

    def next_row(self) -> tuple[str | None, list[float] | None, int]:
        with self._lock:
            if self.mode is None:
                return None, None, self.version
            if self.queue:
                self.last = self.queue.pop(0)
            return self.mode, self.last, self.version

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.state)

    def set_state(self, state: dict[str, Any]) -> None:
        with self._lock:
            self.state = state



class CameraBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.frames: dict[str, bytes] = {}
        self.frame_id = 0

    def set_frame(self, name: str, png_bytes: bytes) -> None:
        with self._lock:
            self.frames[name] = png_bytes
            self.frame_id += 1

    def get_frame(self, name: str) -> bytes | None:
        with self._lock:
            return self.frames.get(name)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {"frame_id": self.frame_id, "available": sorted(self.frames.keys())}

COMMAND_BUFFER = CommandBuffer()
CAMERA_BUFFER = CameraBuffer()


def _abs_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _rows_from_payload(payload: dict[str, Any], key: str, width: int) -> list[list[float]]:
    if key not in payload:
        raise ValueError(f"Missing JSON field: {key}")
    value = payload[key]
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    if len(value) == width and all(isinstance(x, (int, float)) for x in value):
        return [[float(x) for x in value]]
    rows = []
    for row in value:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError(f"Each {key} row must have length {width}")
        rows.append([float(x) for x in row])
    if not rows:
        raise ValueError(f"{key} cannot be empty")
    return rows


class ControlHandler(BaseHTTPRequestHandler):
    server_version = "BRXControlHTTP/0.1"

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
            self._send_json(200, {"ok": True})
        elif self.path == "/camera":
            self._send_json(200, CAMERA_BUFFER.status())
        elif self.path in ("/camera/head.png", "/camera/left_wrist.png", "/camera/right_wrist.png"):
            name = self.path.split("/")[-1].replace(".png", "")
            frame = CAMERA_BUFFER.get_frame(name)
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
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
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
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6, restitution=0.0),
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


def _spawn_scene() -> None:
    ground = sim_utils.GroundPlaneCfg(
        color=(0.72, 0.76, 0.76),
        size=(8.0, 8.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9, restitution=0.0),
    )
    ground.func("/World/defaultGroundPlane", ground)
    _spawn_floor_visuals()
    light = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
    light.func("/World/Light", light)
    if args_cli.no_task_scene:
        return
    sim_utils.create_prim("/World/TaskScene", "Xform")
    table_top_z = 0.61
    _spawn_static_cuboid("/World/TaskScene/TableTop", (0.78, 0.72, 0.055), (0.72, 0.0, table_top_z - 0.0275), (0.48, 0.42, 0.34))
    for name, dx, dy in [("LegFL", 0.31, 0.27), ("LegFR", 0.31, -0.27), ("LegBL", -0.31, 0.27), ("LegBR", -0.31, -0.27)]:
        _spawn_static_cuboid(f"/World/TaskScene/{name}", (0.045, 0.045, table_top_z), (0.72 + dx, dy, table_top_z * 0.5), (0.34, 0.30, 0.25))
    _spawn_bucket("/World/TaskScene/Bucket", (0.82, 0.0, table_top_z))
    cube_size = 0.06
    cube_z = table_top_z + cube_size * 0.5 + 0.003
    _spawn_rigid_cube("/World/TaskScene/BlockRed", cube_size, (0.56, 0.16, cube_z), (0.9, 0.12, 0.08))
    _spawn_rigid_cube("/World/TaskScene/BlockBlue", cube_size, (0.56, -0.16, cube_z), (0.08, 0.22, 0.9))



def _make_cameras() -> Camera:
    """Create three fixed RGB cameras for the X-VLA bridge."""
    sim_utils.create_prim("/World/Cameras", "Xform")
    for name in ["Cam00Head", "Cam01LeftWrist", "Cam02RightWrist"]:
        sim_utils.create_prim(f"/World/Cameras/{name}", "Xform")

    camera_cfg = CameraCfg(
        prim_path="/World/Cameras/Cam.*/CameraSensor",
        update_period=0.0,
        height=args_cli.camera_height,
        width=args_cli.camera_width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=2.0,
            horizontal_aperture=24.0,
            clipping_range=(0.05, 20.0),
        ),
    )
    return Camera(cfg=camera_cfg)


def _configure_camera_poses(camera: Camera, device: str) -> None:
    eyes = torch.tensor(
        [
            [1.35, -1.15, 1.25],
            [0.72, 0.82, 0.78],
            [0.72, -0.82, 0.78],
        ],
        dtype=torch.float32,
        device=device,
    )
    targets = torch.tensor(
        [
            [0.70, 0.00, 0.50],
            [0.64, 0.10, 0.50],
            [0.64, -0.10, 0.50],
        ],
        dtype=torch.float32,
        device=device,
    )
    camera.set_world_poses_from_view(eyes, targets)


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vec by Isaac Lab's wxyz quaternion."""
    q_vec = quat[..., 1:4]
    q_w = quat[..., 0:1]
    uv = torch.cross(q_vec, vec, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return vec + 2.0 * (q_w * uv + uuv)


def _update_camera_poses(
    camera: Camera,
    robot: Articulation,
    head_body_id: int,
    left_ctx: ArmIkContext,
    right_ctx: ArmIkContext,
    device: str,
) -> None:
    """Keep camera order aligned with X-VLA: left_eye/global, left_wrist, right_wrist."""
    head_pose = robot.data.body_state_w[0, head_body_id, 0:7]
    left_pose = robot.data.body_state_w[0, left_ctx.entity_cfg.body_ids[0], 0:7]
    right_pose = robot.data.body_state_w[0, right_ctx.entity_cfg.body_ids[0], 0:7]

    head_offset = torch.tensor(args_cli.head_camera_offset, dtype=torch.float32, device=device)
    head_forward = torch.tensor(args_cli.head_camera_forward, dtype=torch.float32, device=device)
    left_offset = torch.tensor(args_cli.left_wrist_camera_offset, dtype=torch.float32, device=device)
    right_offset = torch.tensor(args_cli.right_wrist_camera_offset, dtype=torch.float32, device=device)
    wrist_forward = torch.tensor(args_cli.wrist_camera_forward, dtype=torch.float32, device=device)

    head_eye = head_pose[0:3] + _quat_apply(head_pose[3:7], head_offset)
    left_eye = left_pose[0:3] + _quat_apply(left_pose[3:7], left_offset)
    right_eye = right_pose[0:3] + _quat_apply(right_pose[3:7], right_offset)
    head_target = head_eye + _quat_apply(head_pose[3:7], head_forward)
    left_target = left_eye + _quat_apply(left_pose[3:7], wrist_forward)
    right_target = right_eye + _quat_apply(right_pose[3:7], wrist_forward)

    eyes = torch.stack([head_eye, left_eye, right_eye], dim=0)
    targets = torch.stack([head_target, left_target, right_target], dim=0)
    camera.set_world_poses_from_view(eyes, targets)


def _rgb_tensor_to_png_bytes(rgb: torch.Tensor) -> bytes:
    array = rgb.detach().cpu().numpy()
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != "uint8":
        if array.max() <= 1.0:
            array = array * 255.0
        array = array.clip(0, 255).astype("uint8")
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _update_camera_cache(camera: Camera) -> None:
    if "rgb" not in camera.data.output:
        return
    rgb = camera.data.output["rgb"]
    if rgb.ndim != 4:
        return
    for index, name in enumerate(CAMERA_NAMES):
        if index < rgb.shape[0]:
            CAMERA_BUFFER.set_frame(name, _rgb_tensor_to_png_bytes(rgb[index]))

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
    return mat[:, 0:2].reshape(-1).detach().cpu().tolist()


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


def _apply_gripper_targets(robot: Articulation, left_grip: float, right_grip: float) -> None:
    left = max(0.0, min(0.041, float(left_grip)))
    right = max(0.0, min(0.041, float(right_grip)))
    names = RIGHT_GRIPPER_JOINTS + LEFT_GRIPPER_JOINTS
    if not all(name in robot.joint_names for name in names):
        return
    joint_ids = [robot.joint_names.index(name) for name in names]
    targets = torch.tensor([[right, -right, left, -left]], device=robot.device)
    robot.set_joint_position_target(targets, joint_ids=joint_ids)


def _joint23_to_full_tensor(robot: Articulation, row: list[float]) -> torch.Tensor:
    target = robot.data.joint_pos.clone()
    for fk_idx, joint_name in enumerate(EXPECTED_MOVABLE_JOINTS):
        if joint_name in robot.joint_names:
            target[:, robot.joint_names.index(joint_name)] = float(row[fk_idx])
    return target


def _apply_joint23(robot: Articulation, row: list[float]) -> None:
    target = _joint23_to_full_tensor(robot, row)
    robot.set_joint_position_target(target)


def _reset_joint23(robot: Articulation, row: list[float]) -> None:
    target = _joint23_to_full_tensor(robot, row)
    joint_vel = torch.zeros_like(target)
    robot.write_joint_state_to_sim(target, joint_vel)
    robot.set_joint_position_target(target)
    robot.reset()


def _apply_ee6d(sim: SimulationContext, robot: Articulation, left_ctx: ArmIkContext, right_ctx: ArmIkContext, row: list[float]) -> None:
    left_pose_b, left_grip = _ee10_to_pose(row[0:10], sim.device)
    right_pose_b, right_grip = _ee10_to_pose(row[10:20], sim.device)
    left_ctx.controller.set_command(left_pose_b)
    right_ctx.controller.set_command(right_pose_b)
    robot.set_joint_position_target(_compute_arm_command(robot, left_ctx), joint_ids=left_ctx.entity_cfg.joint_ids)
    robot.set_joint_position_target(_compute_arm_command(robot, right_ctx), joint_ids=right_ctx.entity_cfg.joint_ids)
    _apply_gripper_targets(robot, left_grip=left_grip, right_grip=right_grip)


def _state_snapshot(robot: Articulation, left_ctx: ArmIkContext, right_ctx: ArmIkContext) -> dict[str, Any]:
    joint_pos = robot.data.joint_pos[0]
    qpos23 = []
    for name in EXPECTED_MOVABLE_JOINTS:
        qpos23.append(float(joint_pos[robot.joint_names.index(name)].detach().cpu()) if name in robot.joint_names else None)

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
    if all(name in robot.joint_names for name in RIGHT_GRIPPER_JOINTS):
        right_ids = [robot.joint_names.index(name) for name in RIGHT_GRIPPER_JOINTS]
        right_grip = float(torch.mean(torch.abs(joint_pos[right_ids])).detach().cpu())
    if all(name in robot.joint_names for name in LEFT_GRIPPER_JOINTS):
        left_ids = [robot.joint_names.index(name) for name in LEFT_GRIPPER_JOINTS]
        left_grip = float(torch.mean(torch.abs(joint_pos[left_ids])).detach().cpu())

    left_ee10_base = left_pos_b[0].detach().cpu().tolist() + _quat_to_rot6d(left_quat_b[0]) + [left_grip]
    right_ee10_base = right_pos_b[0].detach().cpu().tolist() + _quat_to_rot6d(right_quat_b[0]) + [right_grip]

    return {
        "ready": True,
        "mode": COMMAND_BUFFER.mode,
        "qpos23": qpos23,
        "ee6d_base": left_ee10_base + right_ee10_base,
        "left_ee_base": left_ee10_base,
        "right_ee_base": right_ee10_base,
        "left_ee_world": left_pose_w.detach().cpu().tolist(),
        "right_ee_world": right_pose_w.detach().cpu().tolist(),
    }


def run_simulator(sim: SimulationContext, robot: Articulation, camera: Camera) -> None:
    sim_dt = sim.get_physics_dt()
    _configure_camera_poses(camera, sim.device)
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim_dt)
    camera.update(sim_dt)
    _update_camera_cache(camera)

    left_ctx = _resolve_arm(sim, robot, LEFT_ARM_JOINTS, args_cli.left_ee_body)
    right_ctx = _resolve_arm(sim, robot, RIGHT_ARM_JOINTS, args_cli.right_ee_body)
    missing = [name for name in EXPECTED_MOVABLE_JOINTS if name not in robot.joint_names]
    if missing:
        raise RuntimeError(f"Missing expected joint names: {missing}")
    if args_cli.head_camera_body not in robot.body_names:
        raise RuntimeError(f"Missing head camera body: {args_cli.head_camera_body}")
    head_body_id = robot.body_names.index(args_cli.head_camera_body)
    _update_camera_poses(camera, robot, head_body_id, left_ctx, right_ctx, sim.device)
    camera.update(sim_dt)
    _update_camera_cache(camera)

    print("[BRX] Control conventions:")
    print("[BRX] ee6d: [left_xyz, left_rot6d, left_gripper, right_xyz, right_rot6d, right_gripper], absolute base frame")
    print("[BRX] joint23: absolute joint targets in EXPECTED_MOVABLE_JOINTS order")
    print("[BRX] gripper scalar is interpreted as jaw meters and clamped to [0, 0.041]")
    COMMAND_BUFFER.set_state(_state_snapshot(robot, left_ctx, right_ctx))

    hold_count = 0
    current_mode: str | None = None
    current_row: list[float] | None = None
    while simulation_app.is_running():
        if hold_count <= 0:
            current_mode, current_row, _ = COMMAND_BUFFER.next_row()
            hold_count = max(1, args_cli.command_hold_steps)
        hold_count -= 1

        if current_mode == "joint23" and current_row is not None:
            _apply_joint23(robot, current_row)
        elif current_mode == "reset_joint23" and current_row is not None:
            _reset_joint23(robot, current_row)
            COMMAND_BUFFER.set_command("stop", [])
        elif current_mode == "ee6d" and current_row is not None:
            _apply_ee6d(sim, robot, left_ctx, right_ctx, current_row)
        elif current_mode == "stop":
            robot.set_joint_position_target(robot.data.joint_pos)

        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        _update_camera_poses(camera, robot, head_body_id, left_ctx, right_ctx, sim.device)
        camera.update(sim_dt)
        _update_camera_cache(camera)
        COMMAND_BUFFER.set_state(_state_snapshot(robot, left_ctx, right_ctx))


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, -2.5, 2.0], [0.5, 0.0, 0.5])
    _spawn_scene()
    camera = _make_cameras()
    robot = Articulation(cfg=_make_robot_cfg())
    sim.reset()
    server = _start_http_server()
    print("[BRX] Setup complete.")
    try:
        run_simulator(sim, robot, camera)
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()
