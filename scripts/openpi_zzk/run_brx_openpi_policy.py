"""Run a BRX042501 23D openpi policy in Isaac Lab, or replay 23D HDF5 actions.

The script is self-contained and does not import or modify scripts/custom.
"""

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="BRX042501 openpi policy runner for Isaac Lab.")
parser.add_argument("--mode", choices=["policy", "replay_hdf5"], default="policy")
parser.add_argument("--openpi_root", type=str, default="/home/kemove/openpi_zzk")
parser.add_argument("--config_name", type=str, default="pi05_brx_finetune")
parser.add_argument("--checkpoint_dir", type=str, default="")
parser.add_argument("--prompt", type=str, default="move the object smoothly")
parser.add_argument("--action_hdf5", type=str, default="")
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
parser.add_argument("--camera_width", type=int, default=640)
parser.add_argument("--camera_height", type=int, default=360)
parser.add_argument("--camera_pose_mode", choices=["link", "lookat", "fixed"], default="link")
parser.add_argument("--default_head02", type=float, default=-0.17918)
parser.add_argument("--default_head03", type=float, default=-0.81304)
parser.add_argument("--head_camera_body", type=str, default="EyeL_Link")
parser.add_argument("--left_wrist_camera_body", type=str, default="HandCam02_Link")
parser.add_argument("--right_wrist_camera_body", type=str, default="HandCam01_Link")
parser.add_argument("--head_camera_offset", type=float, nargs=3, default=(0.0, 0.0, 0.0))
parser.add_argument("--head_camera_forward", type=float, nargs=3, default=(0.25, 0.0, 0.0))
parser.add_argument("--wrist_camera_forward", type=float, nargs=3, default=(0.20, 0.0, -0.05))
parser.add_argument("--no_task_scene", action="store_true")
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


def _abs_path(path: str) -> str:
    return str(Path(path).expanduser().resolve())


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
            usd_file_name=f"{Path(urdf_path).stem}_openpi_imported.usd",
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
    _spawn_rigid_cube("/World/TaskScene/BlockA", 0.05, (-0.55, 0.0, 2.30), (1.0, 0.5, 0.5))
    _spawn_rigid_cube("/World/TaskScene/BlockB", 0.05, (-0.55, 0.20, 2.30), (0.5, 0.8, 1.0))


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


def _set_fixed_camera_poses(camera: Camera, device: str) -> None:
    eyes = torch.tensor(
        [[1.35, -1.15, 1.25], [0.72, 0.82, 0.78], [0.72, -0.82, 0.78]],
        dtype=torch.float32,
        device=device,
    )
    targets = torch.tensor(
        [[0.70, 0.00, 0.50], [0.64, 0.10, 0.50], [0.64, -0.10, 0.50]],
        dtype=torch.float32,
        device=device,
    )
    camera.set_world_poses_from_view(eyes, targets)


def _update_camera_poses(camera: Camera, robot: Articulation, device: str) -> None:
    if args_cli.camera_pose_mode == "fixed":
        _set_fixed_camera_poses(camera, device)
        return
    body_ids = []
    for body_name in [args_cli.head_camera_body, args_cli.left_wrist_camera_body, args_cli.right_wrist_camera_body]:
        if body_name not in robot.body_names:
            raise RuntimeError(f"Missing camera body: {body_name}")
        body_ids.append(robot.body_names.index(body_name))

    head_pose = robot.data.body_state_w[0, body_ids[0], 0:7]
    left_pose = robot.data.body_state_w[0, body_ids[1], 0:7]
    right_pose = robot.data.body_state_w[0, body_ids[2], 0:7]
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


def _qpos23(robot: Articulation) -> np.ndarray:
    joint_pos = robot.data.joint_pos[0]
    return np.asarray(
        [float(joint_pos[robot.joint_names.index(name)].detach().cpu()) for name in BRX_JOINT_NAMES],
        dtype=np.float32,
    )


def _apply_qpos23(robot: Articulation, row: np.ndarray) -> None:
    if row.shape[-1] != len(BRX_JOINT_NAMES):
        raise ValueError(f"Expected 23D action, got shape {row.shape}")
    target = robot.data.joint_pos.clone()
    for idx, joint_name in enumerate(BRX_JOINT_NAMES):
        target[:, robot.joint_names.index(joint_name)] = float(row[idx])
    robot.set_joint_position_target(target)


def _apply_default_head_pose(robot: Articulation) -> None:
    target = robot.data.joint_pos.clone()
    target[:, robot.joint_names.index("Head02_Joint")] = args_cli.default_head02
    target[:, robot.joint_names.index("Head03_Joint")] = args_cli.default_head03
    robot.write_joint_state_to_sim(target, torch.zeros_like(target))
    robot.set_joint_position_target(target)


def _make_observation(robot: Articulation, camera: Camera) -> dict:
    rgb = camera.data.output["rgb"]
    images = {name: _rgb_to_numpy(rgb[idx]) for idx, name in enumerate(CAMERA_NAMES)}
    images["right_eye"] = images["left_eye"]
    return {
        "state": _qpos23(robot),
        "images": images,
        "prompt": args_cli.prompt,
    }


def _load_policy():
    if not args_cli.checkpoint_dir:
        raise ValueError("--checkpoint_dir is required in policy mode")
    openpi_root = _abs_path(args_cli.openpi_root)
    if openpi_root not in sys.path:
        sys.path.insert(0, openpi_root)
    from openpi.policies import policy_config
    from openpi.training import config as _config

    train_config = _config.get_config(args_cli.config_name)
    return policy_config.create_trained_policy(train_config, _abs_path(args_cli.checkpoint_dir))


def _load_replay_actions() -> np.ndarray:
    if not args_cli.action_hdf5:
        raise ValueError("--action_hdf5 is required in replay_hdf5 mode")
    import h5py

    with h5py.File(_abs_path(args_cli.action_hdf5), "r") as ep:
        actions = np.asarray(ep["/action"][:], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != len(BRX_JOINT_NAMES):
        raise ValueError(f"Expected replay actions shape [T, 23], got {actions.shape}")
    return actions


def _validate_robot(robot: Articulation) -> None:
    missing = [name for name in BRX_JOINT_NAMES if name not in robot.joint_names]
    if missing:
        raise RuntimeError(f"Missing expected BRX joints: {missing}")
    print("[BRX openpi] 23D joint order:")
    for idx, name in enumerate(BRX_JOINT_NAMES):
        print(f"  {idx:02d}: {name}")


def run_simulator(sim: SimulationContext, robot: Articulation, camera: Camera) -> None:
    sim_dt = sim.get_physics_dt()
    _validate_robot(robot)
    _apply_default_head_pose(robot)
    robot.write_data_to_sim()
    for _ in range(max(1, args_cli.warmup_steps)):
        _update_camera_poses(camera, robot, sim.device)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        camera.update(sim_dt)

    policy = _load_policy() if args_cli.mode == "policy" else None
    replay_actions = _load_replay_actions() if args_cli.mode == "replay_hdf5" else None
    action_queue: list[np.ndarray] = []
    replay_idx = 0
    hold_count = 0
    current_action = _qpos23(robot)

    print(f"[BRX openpi] mode={args_cli.mode}, hold_steps={args_cli.command_hold_steps}")
    while simulation_app.is_running():
        if hold_count <= 0:
            if args_cli.mode == "replay_hdf5":
                if replay_idx < len(replay_actions):
                    current_action = replay_actions[replay_idx]
                    replay_idx += 1
                else:
                    current_action = _qpos23(robot)
            else:
                if not action_queue:
                    _update_camera_poses(camera, robot, sim.device)
                    camera.update(sim_dt)
                    result = policy.infer(_make_observation(robot, camera))
                    action_queue = [np.asarray(row, dtype=np.float32) for row in result["actions"]]
                    print(f"[BRX openpi] inferred action chunk: {len(action_queue)} x {action_queue[0].shape[0]}")
                current_action = action_queue.pop(0)
            hold_count = max(1, args_cli.command_hold_steps)

        _apply_qpos23(robot, current_action)
        _update_camera_poses(camera, robot, sim.device)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        camera.update(sim_dt)
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
