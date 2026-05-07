"""Run a BRX042501 23D openpi policy in Isaac Lab, or replay 23D HDF5 actions.

The script is self-contained and does not import or modify scripts/custom.
"""

import argparse
import base64
import io
import json
import os
from pathlib import Path
import random
import time
import urllib.error
import urllib.request

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="BRX042501 openpi policy runner for Isaac Lab.")
parser.add_argument("--mode", choices=["remote_policy", "replay_hdf5"], default="remote_policy")
parser.add_argument("--policy_server_url", type=str, default="http://127.0.0.1:8777/infer")
parser.add_argument("--policy_health_url", type=str, default="", help="Defaults to policy_server_url with /health.")
parser.add_argument("--policy_request_timeout_s", type=float, default=5.0)
parser.add_argument("--policy_poll_s", type=float, default=0.5)
parser.add_argument("--action_hdf5", type=str, default="")
parser.add_argument("--replay_key", choices=["action", "qpos"], default="action", help="HDF5 dataset to replay in replay_hdf5 mode.")
parser.add_argument("--replay_start", type=int, default=0)
parser.add_argument("--replay_end", type=int, default=-1, help="Exclusive end frame. -1 means file end.")
parser.add_argument("--replay_stride", type=int, default=1)
parser.add_argument("--replay_loop", action="store_true")
parser.add_argument("--no_replay_reset_to_first", action="store_true", help="Do not reset robot to the first replay row before playback.")
parser.add_argument("--replay_apply_locks", action="store_true", help="Apply torso/head locks during replay. Default is raw replay for mapping debug.")
parser.add_argument("--initial_qpos23", type=float, nargs=23, default=None, help="Explicit 23D startup qpos in BRX order.")
parser.add_argument("--initial_qpos_hdf5", type=str, default="", help="Optional ACT HDF5 used only when --initial_qpos23 is not provided.")
parser.add_argument("--initial_qpos_frame", type=int, default=0)
parser.add_argument("--no_initial_qpos_reset", action="store_true", help="Disable explicit --initial_qpos23/--initial_qpos_hdf5 reset.")
parser.add_argument("--print_action_summary", action="store_true", help="Print fold/trunk/head values for each new action chunk.")
parser.add_argument("--print_action_table", action="store_true", help="Print full 23D current qpos and first action for each new action chunk.")
parser.add_argument("--save_policy_actions", type=str, default="", help="Optional .npy path to append/save remote policy action chunks for offline comparison.")
parser.add_argument("--urdf_path", type=str, default="/home/kemove/zzk_data/IsaacLab/BRX042501/BRX042501_wheel.urdf")
parser.add_argument("--usd_dir", type=str, default=None)
parser.add_argument("--force_usd_conversion", action="store_true")
parser.add_argument("--no_instanceable", action="store_true")
parser.add_argument("--robot_prim", type=str, default="/World/Robot")
parser.add_argument("--sim_dt", type=float, default=1.0 / 30.0)
parser.add_argument("--command_hold_steps", type=int, default=3)
parser.add_argument("--warmup_steps", type=int, default=30)
parser.add_argument("--joint_stiffness", type=float, default=2500.0)
parser.add_argument("--joint_damping", type=float, default=120.0)
parser.add_argument("--effort_limit", type=float, default=800.0)
parser.add_argument("--velocity_limit", type=float, default=60.0)
parser.add_argument("--teleop_assets_root", type=str, default="teleop/assets")
parser.add_argument("--fixed_blocks", dest="randomize_blocks", action="store_false")
parser.add_argument("--block_seed", type=int, default=None)
parser.add_argument("--camera_width", type=int, default=640)
parser.add_argument("--camera_height", type=int, default=360)
parser.add_argument("--camera_update_every", type=int, default=1)
parser.add_argument("--camera_pose_mode", choices=["link", "lookat"], default="link")
parser.add_argument("--camera_snapshot_dir", type=str, default="/tmp/brx_openpi_camera_snapshots")
parser.add_argument("--camera_snapshot_interval_s", type=float, default=0.0, help="Save 3-camera PNG snapshots every N seconds after startup. 0 saves startup only.")
parser.add_argument("--no_camera_snapshot", action="store_true", help="Disable startup camera snapshot saving.")
parser.add_argument("--default_head02", type=float, default=-0.17918)
parser.add_argument("--default_head03", type=float, default=-0.81304)
parser.add_argument("--startup_arm_qpos_hdf5", type=str, default="/home/kemove/ACT_Datasets/episode_0.hdf5", help="Use arm/gripper qpos from this ACT episode at startup if readable.")
parser.add_argument("--startup_arm_qpos_frame", type=int, default=0)
parser.add_argument("--no_startup_arm_qpos_hdf5", action="store_true", help="Disable dataset-based startup arm pose.")
parser.add_argument("--startup_gripper_open_m", type=float, default=0.041, help="Startup jaw opening target in meters.")
parser.add_argument("--gripper_max_m", type=float, default=0.041, help="Clamp BRX JawBlock policy targets to [0, gripper_max_m].")
parser.add_argument("--no_gripper_clamp", action="store_true", help="Send raw policy gripper values without clamping.")
parser.add_argument("--swap_left_right_gripper", action="store_true", help="Deprecated compatibility flag. Grippers are swapped by default to match ACT data order.")
parser.add_argument("--no_swap_left_right_gripper", action="store_true", help="Disable default ACT-data gripper remap for debugging raw URDF JawBlock order.")
parser.add_argument("--head_camera_body", type=str, default="EyeL_Link")
parser.add_argument("--left_wrist_camera_body", type=str, default="HandCam02_Link")
parser.add_argument("--right_wrist_camera_body", type=str, default="HandCam01_Link")
parser.add_argument("--head_camera_offset", type=float, nargs=3, default=(0.0, 0.0, 0.0))
parser.add_argument("--head_camera_forward", type=float, nargs=3, default=(0.25, 0.0, 0.0))
parser.add_argument("--wrist_camera_forward", type=float, nargs=3, default=(0.20, 0.0, -0.12))
parser.add_argument("--no_task_scene", action="store_true")
parser.add_argument("--policy_start_delay_s", type=float, default=0.0, help="Optional delay before policy/replay starts.")
parser.add_argument("--debug_pose_print_s", type=float, default=1.0, help="Print BRX torso/head/root pose every N seconds. Use 0 to disable.")
parser.add_argument("--unlock_torso", action="store_true", help="Allow policy/replay to command Folding02/Folding03/Trunk. Default locks them upright.")
parser.add_argument("--unlock_head", action="store_true", help="Allow policy/replay to command Head02/Head03. Default locks head to startup pose.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import Camera
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.utils.assets import check_file_path


BRX_JOINT_NAMES = [
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

CAMERA_NAMES = ["left_eye", "left_wrist", "right_wrist"]

DEFAULT_TORSO_QPOS = {
    "FoldingModularJoint02_Joint": 0.0,
    "FoldingModularJoint03_Joint": 0.0,
    "Trunk_Joint": 0.0,
}
ARM_AND_GRIPPER_QPOS23_INDICES = list(range(3, 21))
RIGHT_GRIPPER_JOINTS = ["JawBlock01_Joint", "JawBlock02_Joint"]
LEFT_GRIPPER_JOINTS = ["JawBlock03_Joint", "JawBlock04_Joint"]
GRIPPER_JOINT_NAMES = RIGHT_GRIPPER_JOINTS + LEFT_GRIPPER_JOINTS
GRIPPER_QPOS23_INDICES = [BRX_JOINT_NAMES.index(name) for name in GRIPPER_JOINT_NAMES]
# ACT HDF5 rows store gripper values by arm side: indices 10/11 follow ArmL,
# indices 19/20 follow ArmR. The BRX URDF names are opposite by JawBlock id
# (JawBlock03/04 are the left gripper, JawBlock01/02 are the right gripper), so
# simulation commands remap those two pairs by default.


def _abs_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _asset_path(*parts: str) -> str:
    root = args_cli.teleop_assets_root
    if not os.path.isabs(root):
        root = os.path.join(os.getcwd(), root)
    return _abs_path(os.path.join(root, *parts))


def _make_robot_cfg() -> ArticulationCfg:
    urdf_path = _abs_path(args_cli.urdf_path)
    if not check_file_path(urdf_path):
        raise FileNotFoundError(f"URDF path does not exist or is not readable: {urdf_path}")
    usd_dir = _abs_path(args_cli.usd_dir) if args_cli.usd_dir else str(Path(urdf_path).parent / "isaaclab_converted")
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
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=args_cli.joint_stiffness,
                    damping=args_cli.joint_damping,
                ),
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


def _make_cameras() -> Camera:
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
            focal_length=12.0,
            focus_distance=2.0,
            horizontal_aperture=24.0,
            clipping_range=(0.05, 20.0),
        ),
    )
    return Camera(cfg=camera_cfg)


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    q_vec = quat[..., 1:4]
    q_w = quat[..., 0:1]
    uv = torch.cross(q_vec, vec, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return vec + 2.0 * (q_w * uv + uuv)


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


def _update_camera_poses(
    camera: Camera,
    robot: Articulation,
    head_body_id: int,
    left_wrist_camera_body_id: int,
    right_wrist_camera_body_id: int,
    device: str,
) -> None:
    """Keep camera order aligned with custom BRX: left_eye/global, left_wrist, right_wrist."""
    head_pose = robot.data.body_state_w[0, head_body_id, 0:7]
    left_pose = robot.data.body_state_w[0, left_wrist_camera_body_id, 0:7]
    right_pose = robot.data.body_state_w[0, right_wrist_camera_body_id, 0:7]

    if args_cli.camera_pose_mode == "link":
        positions = torch.stack([head_pose[0:3], left_pose[0:3], right_pose[0:3]], dim=0)
        orientations = torch.stack([head_pose[3:7], left_pose[3:7], right_pose[3:7]], dim=0)
        camera.set_world_poses(positions, orientations, convention="world")
        return

    head_offset = torch.tensor(args_cli.head_camera_offset, dtype=torch.float32, device=device)
    head_forward = torch.tensor(args_cli.head_camera_forward, dtype=torch.float32, device=device)
    wrist_forward = torch.tensor(args_cli.wrist_camera_forward, dtype=torch.float32, device=device)
    head_eye = head_pose[0:3] + _quat_apply(head_pose[3:7], head_offset)
    left_eye = left_pose[0:3]
    right_eye = right_pose[0:3]
    eyes = torch.stack([head_eye, left_eye, right_eye], dim=0)
    targets = torch.stack(
        [
            head_eye + _quat_apply(head_pose[3:7], head_forward),
            left_eye + _quat_apply(left_pose[3:7], wrist_forward),
            right_eye + _quat_apply(right_pose[3:7], wrist_forward),
        ],
        dim=0,
    )
    camera.set_world_poses_from_view(eyes, targets)


def _rgb_to_numpy(rgb: torch.Tensor) -> np.ndarray:
    array = rgb.detach().cpu().numpy()
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = array * 255.0
        array = array.clip(0, 255).astype(np.uint8)
    return array


def _png_b64(array: np.ndarray) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _qpos23(robot: Articulation) -> np.ndarray:
    joint_pos = robot.data.joint_pos[0]
    return np.asarray(
        [float(joint_pos[robot.joint_names.index(name)].detach().cpu()) for name in BRX_JOINT_NAMES],
        dtype=np.float32,
    )


def _apply_qpos23(robot: Articulation, row: np.ndarray) -> None:
    if row.shape[-1] != len(BRX_JOINT_NAMES):
        raise ValueError(f"Expected 23D action, got shape {row.shape}")
    row = _sanitize_qpos23(row)
    target = robot.data.joint_pos.clone()
    for idx, joint_name in enumerate(BRX_JOINT_NAMES):
        target[:, robot.joint_names.index(joint_name)] = float(row[idx])
    robot.set_joint_position_target(target)


def _joint23_to_full_tensor(robot: Articulation, row: np.ndarray) -> torch.Tensor:
    if row.shape[-1] != len(BRX_JOINT_NAMES):
        raise ValueError(f"Expected 23D qpos, got shape {row.shape}")
    row = _sanitize_qpos23(row)
    target = robot.data.joint_pos.clone()
    for idx, joint_name in enumerate(BRX_JOINT_NAMES):
        target[:, robot.joint_names.index(joint_name)] = float(row[idx])
    return target


def _reset_joint23(robot: Articulation, row: np.ndarray) -> None:
    target = _joint23_to_full_tensor(robot, np.asarray(row, dtype=np.float32))
    joint_vel = torch.zeros_like(target)
    robot.write_joint_state_to_sim(target, joint_vel)
    robot.set_joint_position_target(target)
    robot.reset()


def _load_initial_qpos23() -> np.ndarray | None:
    if args_cli.no_initial_qpos_reset:
        return None
    if args_cli.initial_qpos23 is not None:
        qpos = np.asarray(args_cli.initial_qpos23, dtype=np.float32)
        print(f"[BRX openpi] initial qpos reset from --initial_qpos23: fold/trunk/head={np.round(qpos[[0, 1, 2, 21, 22]], 4).tolist()}")
        return qpos
    if not args_cli.initial_qpos_hdf5:
        return None
    path = _abs_path(args_cli.initial_qpos_hdf5)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Initial qpos HDF5 not found: {path}. "
            "Pass --initial_qpos23, pass --initial_qpos_hdf5 to a valid ACT episode, or use --no_initial_qpos_reset."
        )
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required for --initial_qpos_hdf5 reset inside Isaac Lab.") from exc

    with h5py.File(path, "r") as ep:
        qpos = np.asarray(ep["/observations/qpos"][args_cli.initial_qpos_frame], dtype=np.float32)
    if qpos.shape[-1] != len(BRX_JOINT_NAMES):
        raise ValueError(f"Expected initial qpos dim {len(BRX_JOINT_NAMES)}, got {qpos.shape} from {path}")
    print(
        f"[BRX openpi] initial qpos reset from {path} frame={args_cli.initial_qpos_frame}: "
        f"fold/trunk/head={np.round(qpos[[0, 1, 2, 21, 22]], 4).tolist()}"
    )
    return qpos


def _load_startup_arm_qpos23() -> np.ndarray | None:
    if args_cli.no_startup_arm_qpos_hdf5 or not args_cli.startup_arm_qpos_hdf5:
        return None
    path = _abs_path(args_cli.startup_arm_qpos_hdf5)
    if not os.path.exists(path):
        print(f"[BRX openpi] startup arm qpos file not found, using URDF arm pose: {path}")
        return None
    try:
        import h5py
    except ImportError:
        print("[BRX openpi] h5py unavailable, using URDF arm pose")
        return None

    with h5py.File(path, "r") as ep:
        qpos = np.asarray(ep["/observations/qpos"][args_cli.startup_arm_qpos_frame], dtype=np.float32)
    if qpos.shape[-1] != len(BRX_JOINT_NAMES):
        raise ValueError(f"Expected startup arm qpos dim {len(BRX_JOINT_NAMES)}, got {qpos.shape} from {path}")
    print(
        f"[BRX openpi] startup arms/grippers from {path} frame={args_cli.startup_arm_qpos_frame}: "
        f"left_arm={np.round(qpos[3:10], 4).tolist()} "
        f"right_arm={np.round(qpos[12:19], 4).tolist()} "
        f"dataset_grippers={np.round(qpos[[10, 11, 19, 20]], 4).tolist()}"
    )
    return qpos


def _print_action_summary(source: str, rows: list[np.ndarray]) -> None:
    if not args_cli.print_action_summary or not rows:
        return
    row = np.asarray(rows[0], dtype=np.float32)
    print(
        f"[BRX openpi] {source} first action fold/trunk/head="
        f"{np.round(row[[0, 1, 2, 21, 22]], 4).tolist()}"
    )


def _sanitize_qpos23(row: np.ndarray, clamp_gripper: bool | None = None) -> np.ndarray:
    safe = np.asarray(row, dtype=np.float32).copy()
    if _use_act_gripper_remap():
        right = safe[[10, 11]].copy()
        left = safe[[19, 20]].copy()
        safe[[10, 11]] = left
        safe[[19, 20]] = right
    if clamp_gripper is None:
        clamp_gripper = args_cli.mode == "remote_policy" and not args_cli.no_gripper_clamp
    if clamp_gripper:
        safe[GRIPPER_QPOS23_INDICES] = np.clip(safe[GRIPPER_QPOS23_INDICES], 0.0, args_cli.gripper_max_m)
    return safe


def _use_act_gripper_remap() -> bool:
    return args_cli.swap_left_right_gripper or not args_cli.no_swap_left_right_gripper


def _print_action_table(source: str, current_qpos: np.ndarray, rows: list[np.ndarray]) -> None:
    if not args_cli.print_action_table or not rows:
        return
    raw_first = np.asarray(rows[0], dtype=np.float32)
    first = _sanitize_qpos23(raw_first)
    delta = first - np.asarray(current_qpos, dtype=np.float32)
    print(f"[BRX action table] {source} first action vs current qpos")
    for idx, name in enumerate(BRX_JOINT_NAMES):
        print(
            f"  {idx:02d} {name}: "
            f"qpos={current_qpos[idx]: .5f} action={first[idx]: .5f} delta={delta[idx]: .5f}"
        )
    print(
        "[BRX action table] first chunk range: "
        f"raw_min={float(np.min(raw_first)):.5f} raw_max={float(np.max(raw_first)):.5f} "
        f"applied_min={float(np.min(first)):.5f} applied_max={float(np.max(first)):.5f} "
        f"dataset_left_grip_raw_idx10_11={np.round(raw_first[10:12], 5).tolist()} "
        f"dataset_right_grip_raw_idx19_20={np.round(raw_first[19:21], 5).tolist()} "
        f"urdf_right_grip_applied_idx10_11={np.round(first[10:12], 5).tolist()} "
        f"urdf_left_grip_applied_idx19_20={np.round(first[19:21], 5).tolist()}"
    )


def _save_policy_action_chunks(chunks: list[np.ndarray], new_rows: list[np.ndarray]) -> list[np.ndarray]:
    if not args_cli.save_policy_actions or not new_rows:
        return chunks
    chunks.extend(np.asarray(row, dtype=np.float32).copy() for row in new_rows)
    path = _abs_path(args_cli.save_policy_actions)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, np.stack(chunks, axis=0))
    return chunks


def _debug_pose_snapshot(robot: Articulation, label: str) -> None:
    if args_cli.debug_pose_print_s <= 0.0:
        return
    joint_pos = robot.data.joint_pos[0]
    values = {}
    for name in [
        "FoldingModularJoint02_Joint",
        "FoldingModularJoint03_Joint",
        "Trunk_Joint",
        "Head02_Joint",
        "Head03_Joint",
    ]:
        if name in robot.joint_names:
            values[name] = round(float(joint_pos[robot.joint_names.index(name)].detach().cpu()), 5)
        else:
            values[name] = None

    root = robot.data.root_pose_w[0].detach().cpu().numpy()
    body_heights = {}
    for name in [
        "Base_Link",
        "FoldingModule02_Link",
        "FoldingModule03_Link",
        "Trunk_Link",
        "Head02_Link",
        "Head03_Link",
        args_cli.head_camera_body,
    ]:
        if name in robot.body_names:
            body_heights[name] = round(float(robot.data.body_state_w[0, robot.body_names.index(name), 2].detach().cpu()), 4)

    print(
        f"[BRX pose] {label} "
        f"joints={values} "
        f"root_xyz={np.round(root[0:3], 4).tolist()} "
        f"root_quat_wxyz={np.round(root[3:7], 4).tolist()} "
        f"body_z={body_heights}"
    )


def _lock_non_arm_joints(row: np.ndarray, locked_qpos: np.ndarray) -> np.ndarray:
    """Keep the robot upright by default.

    BRX tabletop manipulation should not let a small LoRA policy freely command
    the folding mast/trunk. Head is also fixed to keep the camera distribution
    stable unless explicitly unlocked.
    """
    safe = np.asarray(row, dtype=np.float32).copy()
    if not args_cli.unlock_torso:
        safe[[0, 1, 2]] = locked_qpos[[0, 1, 2]]
    if not args_cli.unlock_head:
        safe[[21, 22]] = locked_qpos[[21, 22]]
    return safe


def _default_locked_qpos23(robot: Articulation) -> np.ndarray:
    qpos = _qpos23(robot)
    qpos[[0, 1, 2]] = 0.0
    qpos[21] = args_cli.default_head02
    qpos[22] = args_cli.default_head03
    return qpos


def _hold_locked_pose(robot: Articulation, locked_qpos: np.ndarray | None = None) -> None:
    if locked_qpos is None:
        robot.set_joint_position_target(robot.data.joint_pos)
        return
    target = _joint23_to_full_tensor(robot, _lock_non_arm_joints(_qpos23(robot), locked_qpos))
    robot.set_joint_position_target(target)


def _apply_default_startup_pose(robot: Articulation, startup_arm_qpos: np.ndarray | None) -> None:
    qpos = _qpos23(robot)
    if startup_arm_qpos is not None:
        qpos[ARM_AND_GRIPPER_QPOS23_INDICES] = np.asarray(startup_arm_qpos, dtype=np.float32)[ARM_AND_GRIPPER_QPOS23_INDICES]
    else:
        startup_grip = max(0.0, min(args_cli.gripper_max_m, args_cli.startup_gripper_open_m))
        qpos[GRIPPER_QPOS23_INDICES] = startup_grip
    qpos[[0, 1, 2]] = 0.0
    qpos[21] = args_cli.default_head02
    qpos[22] = args_cli.default_head03
    qpos = _sanitize_qpos23(qpos)

    names = BRX_JOINT_NAMES
    if not all(name in robot.joint_names for name in names):
        missing = [name for name in names if name not in robot.joint_names]
        raise RuntimeError(f"Missing startup joint(s): {missing}")
    joint_ids = [robot.joint_names.index(name) for name in names]
    values = torch.tensor([qpos.tolist()], dtype=torch.float32, device=robot.device)

    joint_pos = robot.data.joint_pos.clone()
    joint_vel = robot.data.joint_vel.clone()
    joint_pos[:, joint_ids] = values
    joint_vel[:, joint_ids] = 0.0
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(values, joint_ids=joint_ids)
    print(
        "[BRX openpi] startup qpos applied: "
        f"fold/trunk/head={np.round(qpos[[0, 1, 2, 21, 22]], 4).tolist()} "
        f"left_arm={np.round(qpos[3:10], 4).tolist()} "
        f"right_arm={np.round(qpos[12:19], 4).tolist()} "
        f"urdf_grippers={np.round(qpos[[10, 11, 19, 20]], 4).tolist()}"
    )


def _make_observation(robot: Articulation, camera: Camera) -> dict:
    rgb = camera.data.output["rgb"]
    images = {name: _rgb_to_numpy(rgb[idx]) for idx, name in enumerate(CAMERA_NAMES)}
    images["right_eye"] = images["left_eye"]
    return {
        "state": _qpos23(robot),
        "images": images,
    }


def _save_camera_snapshots(camera: Camera, label: str, step_count: int = 0) -> None:
    if args_cli.no_camera_snapshot:
        return
    from PIL import Image

    out_dir = _abs_path(args_cli.camera_snapshot_dir)
    os.makedirs(out_dir, exist_ok=True)

    rgb = camera.data.output["rgb"]
    images = {name: _rgb_to_numpy(rgb[idx]) for idx, name in enumerate(CAMERA_NAMES)}
    safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)
    prefix = f"{step_count:06d}_{safe_label}"
    paths = []
    for name, image in images.items():
        path = os.path.join(out_dir, f"{prefix}_{name}.png")
        Image.fromarray(image).save(path)
        paths.append(path)

    height = max(image.shape[0] for image in images.values())
    width = sum(image.shape[1] for image in images.values())
    contact = Image.new("RGB", (width, height))
    x_offset = 0
    for name in CAMERA_NAMES:
        image = Image.fromarray(images[name])
        contact.paste(image, (x_offset, 0))
        x_offset += image.width
    contact_path = os.path.join(out_dir, f"{prefix}_contact_sheet.png")
    contact.save(contact_path)
    print(f"[BRX camera] saved {len(paths)} views + contact sheet: {contact_path}")


def _load_replay_rows() -> np.ndarray:
    if not args_cli.action_hdf5:
        raise ValueError("--action_hdf5 is required in replay_hdf5 mode")
    import h5py

    dataset_path = "/action" if args_cli.replay_key == "action" else "/observations/qpos"
    with h5py.File(_abs_path(args_cli.action_hdf5), "r") as ep:
        if dataset_path not in ep:
            raise KeyError(f"Replay dataset not found in {args_cli.action_hdf5}: {dataset_path}")
        rows = np.asarray(ep[dataset_path][:], dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != len(BRX_JOINT_NAMES):
        raise ValueError(f"Expected replay rows shape [T, 23], got {rows.shape} from {dataset_path}")

    start = max(0, args_cli.replay_start)
    end = len(rows) if args_cli.replay_end < 0 else min(len(rows), args_cli.replay_end)
    stride = max(1, args_cli.replay_stride)
    rows = rows[start:end:stride]
    if len(rows) == 0:
        raise ValueError(f"Replay slice is empty: start={start}, end={end}, stride={stride}")
    print(
        f"[BRX replay] loaded {len(rows)} rows from {args_cli.action_hdf5}:{dataset_path} "
        f"slice=[{start}:{end}:{stride}]"
    )
    print(
        "[BRX replay] first row fold/trunk/head/gripper="
        f"{np.round(rows[0][[0, 1, 2, 21, 22, 10, 11, 19, 20]], 5).tolist()}"
    )
    print(
        "[BRX replay] gripper raw range "
        f"dataset_left_idx10_11={np.round([np.min(rows[:, 10:12]), np.max(rows[:, 10:12])], 5).tolist()} "
        f"dataset_right_idx19_20={np.round([np.min(rows[:, 19:21]), np.max(rows[:, 19:21])], 5).tolist()} "
        f"act_gripper_remap={_use_act_gripper_remap()}"
    )
    return rows


def _infer_remote_policy(observation: dict) -> list[np.ndarray]:
    images = observation["images"]
    payload = {
        "state": np.asarray(observation["state"], dtype=np.float32).tolist(),
        "images": {
            "left_eye": _png_b64(images["left_eye"]),
            "left_wrist": _png_b64(images["left_wrist"]),
            "right_wrist": _png_b64(images["right_wrist"]),
        },
    }
    request = urllib.request.Request(
        args_cli.policy_server_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args_cli.policy_request_timeout_s) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not result.get("ok", False):
        raise RuntimeError(f"Remote policy error: {result}")
    return [np.asarray(row, dtype=np.float32) for row in result["actions"]]


def _policy_health_url() -> str:
    if args_cli.policy_health_url:
        return args_cli.policy_health_url
    infer_url = args_cli.policy_server_url.rstrip("/")
    if infer_url.endswith("/infer"):
        return f"{infer_url[:-len('/infer')]}/health"
    return f"{infer_url}/health"


def _policy_server_ready(timeout_s: float = 0.3) -> bool:
    request = urllib.request.Request(_policy_health_url(), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
            return bool(payload.get("ok", False))
    except (TimeoutError, urllib.error.URLError, json.JSONDecodeError):
        return False


def _validate_robot(robot: Articulation) -> None:
    missing = [name for name in BRX_JOINT_NAMES if name not in robot.joint_names]
    if missing:
        raise RuntimeError(f"Missing expected BRX joints: {missing}")
    print("[BRX openpi] 23D joint order:")
    for idx, name in enumerate(BRX_JOINT_NAMES):
        print(f"  {idx:02d}: {name}")


def _render_viewport(sim: SimulationContext) -> None:
    """Force a viewport render for livestream/WebRTC in headless runs."""
    try:
        sim.render()
    except Exception:
        # Older Isaac Lab builds may render implicitly during sim.step().
        pass


def _hold_until_policy_start(
    sim: SimulationContext,
    robot: Articulation,
    camera: Camera,
    head_body_id: int,
    left_wrist_camera_body_id: int,
    right_wrist_camera_body_id: int,
    sim_dt: float,
    locked_qpos: np.ndarray | None,
) -> None:
    if args_cli.mode != "remote_policy" and args_cli.policy_start_delay_s <= 0.0:
        return

    start_time = time.monotonic()
    last_wait_print = 0.0
    last_policy_poll = 0.0
    last_pose_print = time.monotonic()
    policy_ready = False
    if args_cli.mode == "remote_policy":
        print(f"[BRX openpi] waiting for policy server health: {_policy_health_url()}")
    if args_cli.policy_start_delay_s > 0.0:
        print(f"[BRX openpi] delaying policy start for {args_cli.policy_start_delay_s:.1f}s")

    while simulation_app.is_running():
        now = time.monotonic()
        if args_cli.mode == "remote_policy" and now - last_policy_poll >= max(0.1, args_cli.policy_poll_s):
            policy_ready = _policy_server_ready(timeout_s=min(0.2, args_cli.policy_request_timeout_s))
            last_policy_poll = now

        delay_done = time.monotonic() - start_time >= args_cli.policy_start_delay_s
        policy_done = args_cli.mode != "remote_policy" or policy_ready
        if delay_done and policy_done:
            print("[BRX openpi] policy server ready; starting actions")
            return

        if args_cli.mode == "remote_policy" and now - last_wait_print >= 2.0:
            print("[BRX openpi] sim is rendering; policy server not ready yet")
            last_wait_print = now
        if args_cli.debug_pose_print_s > 0.0 and now - last_pose_print >= args_cli.debug_pose_print_s:
            _debug_pose_snapshot(robot, "waiting_for_policy")
            last_pose_print = now

        _hold_locked_pose(robot, locked_qpos)
        _update_camera_poses(camera, robot, head_body_id, left_wrist_camera_body_id, right_wrist_camera_body_id, sim.device)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        camera.update(sim_dt)
        _render_viewport(sim)


def run_simulator(sim: SimulationContext, robot: Articulation, camera: Camera) -> None:
    sim_dt = sim.get_physics_dt()
    _validate_robot(robot)
    replay_rows = _load_replay_rows() if args_cli.mode == "replay_hdf5" else None
    initial_qpos = _load_initial_qpos23()
    if args_cli.mode == "replay_hdf5" and initial_qpos is None and not args_cli.no_replay_reset_to_first:
        initial_qpos = np.asarray(replay_rows[0], dtype=np.float32)
        print("[BRX replay] resetting robot to first replay row before playback")
    if initial_qpos is not None:
        _reset_joint23(robot, initial_qpos)
        locked_qpos = np.asarray(initial_qpos, dtype=np.float32).copy()
    else:
        _apply_default_startup_pose(robot, _load_startup_arm_qpos23())
        locked_qpos = _default_locked_qpos23(robot)
    _debug_pose_snapshot(robot, "after_startup_pose_write")
    _configure_camera_poses(camera, sim.device)
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim_dt)
    camera.update(sim_dt)
    _render_viewport(sim)
    _debug_pose_snapshot(robot, "after_first_step")

    if args_cli.head_camera_body not in robot.body_names:
        raise RuntimeError(f"Missing head camera body: {args_cli.head_camera_body}")
    head_body_id = robot.body_names.index(args_cli.head_camera_body)
    if args_cli.left_wrist_camera_body not in robot.body_names:
        raise RuntimeError(f"Missing left wrist camera body: {args_cli.left_wrist_camera_body}")
    if args_cli.right_wrist_camera_body not in robot.body_names:
        raise RuntimeError(f"Missing right wrist camera body: {args_cli.right_wrist_camera_body}")
    left_wrist_camera_body_id = robot.body_names.index(args_cli.left_wrist_camera_body)
    right_wrist_camera_body_id = robot.body_names.index(args_cli.right_wrist_camera_body)
    print(f"[BRX] head camera body: {args_cli.head_camera_body} -> body_index={head_body_id}")
    print(f"[BRX] left wrist camera body: {args_cli.left_wrist_camera_body} -> body_index={left_wrist_camera_body_id}")
    print(f"[BRX] right wrist camera body: {args_cli.right_wrist_camera_body} -> body_index={right_wrist_camera_body_id}")
    _update_camera_poses(camera, robot, head_body_id, left_wrist_camera_body_id, right_wrist_camera_body_id, sim.device)
    camera_update_every = max(1, args_cli.camera_update_every)
    camera.update(sim_dt * camera_update_every)
    for _ in range(max(1, args_cli.warmup_steps)):
        _hold_locked_pose(robot, locked_qpos)
        _update_camera_poses(camera, robot, head_body_id, left_wrist_camera_body_id, right_wrist_camera_body_id, sim.device)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        camera.update(sim_dt)
        _render_viewport(sim)
    _debug_pose_snapshot(robot, "after_warmup")
    _save_camera_snapshots(camera, "after_warmup", 0)

    action_queue: list[np.ndarray] = []
    saved_policy_actions: list[np.ndarray] = []
    replay_idx = 0
    hold_count = 0
    step_count = 0
    current_action = _qpos23(robot)
    print(
        "[BRX openpi] locked startup fold/trunk/head="
        f"{np.round(locked_qpos[[0, 1, 2, 21, 22]], 4).tolist()} "
        f"(unlock_torso={args_cli.unlock_torso}, unlock_head={args_cli.unlock_head})"
    )

    print(f"[BRX openpi] mode={args_cli.mode}, hold_steps={args_cli.command_hold_steps}")
    last_pose_print = time.monotonic()
    last_camera_snapshot = time.monotonic()
    _hold_until_policy_start(
        sim,
        robot,
        camera,
        head_body_id,
        left_wrist_camera_body_id,
        right_wrist_camera_body_id,
        sim_dt,
        locked_qpos,
    )
    while simulation_app.is_running():
        now = time.monotonic()
        if args_cli.debug_pose_print_s > 0.0 and now - last_pose_print >= args_cli.debug_pose_print_s:
            _debug_pose_snapshot(robot, "run_loop")
            last_pose_print = now
        if args_cli.camera_snapshot_interval_s > 0.0 and now - last_camera_snapshot >= args_cli.camera_snapshot_interval_s:
            _save_camera_snapshots(camera, "run_loop", step_count)
            last_camera_snapshot = now
        if hold_count <= 0:
            if args_cli.mode == "replay_hdf5":
                if replay_idx < len(replay_rows):
                    current_action = replay_rows[replay_idx]
                    if replay_idx == 0 or (args_cli.print_action_table and replay_idx % 30 == 0):
                        _print_action_table(f"replay {args_cli.replay_key} frame={replay_idx}", _qpos23(robot), [current_action])
                    replay_idx += 1
                elif args_cli.replay_loop:
                    replay_idx = 0
                    current_action = replay_rows[replay_idx]
                    print("[BRX replay] loop restart")
                    replay_idx += 1
                else:
                    current_action = _qpos23(robot)
            else:
                if not action_queue:
                    _update_camera_poses(camera, robot, head_body_id, left_wrist_camera_body_id, right_wrist_camera_body_id, sim.device)
                    camera.update(sim_dt)
                    if not _policy_server_ready():
                        current_action = _qpos23(robot)
                    else:
                        try:
                            obs = _make_observation(robot, camera)
                            action_queue = _infer_remote_policy(obs)
                            print(f"[BRX openpi] remote action chunk: {len(action_queue)} x {action_queue[0].shape[0]}")
                            _print_action_summary("remote policy", action_queue)
                            _print_action_table("remote policy", obs["state"], action_queue)
                            saved_policy_actions = _save_policy_action_chunks(saved_policy_actions, action_queue)
                        except (TimeoutError, urllib.error.URLError, RuntimeError) as exc:
                            print(f"[BRX openpi] remote policy unavailable; holding current pose: {exc}")
                            action_queue = []
                            current_action = _qpos23(robot)
                if action_queue:
                    current_action = action_queue.pop(0)
            hold_count = max(1, args_cli.command_hold_steps)

        if args_cli.mode == "remote_policy" or args_cli.replay_apply_locks:
            current_action = _lock_non_arm_joints(current_action, locked_qpos)
        _apply_qpos23(robot, current_action)
        _update_camera_poses(camera, robot, head_body_id, left_wrist_camera_body_id, right_wrist_camera_body_id, sim.device)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        _render_viewport(sim)
        step_count += 1
        if step_count % camera_update_every == 0:
            _update_camera_poses(camera, robot, head_body_id, left_wrist_camera_body_id, right_wrist_camera_body_id, sim.device)
            camera.update(sim_dt * camera_update_every)
        hold_count -= 1


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.sim_dt, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.0, 1.0, 2.0], [0.0, 0.0, 1.0])
    _spawn_scene()
    camera = _make_cameras()
    robot = Articulation(cfg=_make_robot_cfg())
    sim.reset()
    print("[BRX openpi] setup complete")
    run_simulator(sim, robot, camera)


if __name__ == "__main__":
    main()
    simulation_app.close()
