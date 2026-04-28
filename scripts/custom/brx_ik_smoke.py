# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""
BRX URDF -> Isaac Lab Differential IK smoke test.

This script does not assume that you already have a clean robot USD or a complete
scene USD. It imports the BRX robot directly from URDF, spawns a minimal Isaac Lab
scene, prints the resolved joint/body names, checks the names used by the X-VLA FK
adapter, and drives the left/right end-effectors through small absolute pose targets.

Usage examples:

    ./isaaclab.sh -p scripts/custom/brx_ik_smoke.py --print_only

    ./isaaclab.sh -p scripts/custom/brx_ik_smoke.py \
        --urdf_path /home/kemove/zzk_data/IsaacLab/BRX042501_wheel.urdf

The first stage intentionally does not call X-VLA. It verifies that the URDF import,
articulation, body names, joint names, Jacobians, and Differential IK control path
are usable before policy integration.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="BRX URDF Differential IK smoke test.")
parser.add_argument(
    "--urdf_path",
    type=str,
    default="/home/kemove/zzk_data/IsaacLab/BRX042501_wheel.urdf",
    help="Path to BRX042501_wheel.urdf. The URDF mesh paths must be valid from this file.",
)
parser.add_argument(
    "--usd_dir",
    type=str,
    default=None,
    help="Directory where Isaac Lab writes the converted USD. Defaults to <urdf_dir>/isaaclab_converted.",
)
parser.add_argument(
    "--force_usd_conversion",
    action="store_true",
    help="Force URDF -> USD conversion even if a converted USD already exists.",
)
parser.add_argument(
    "--no_instanceable",
    action="store_true",
    help="Disable instanceable USD generation. Useful when debugging invisible imported meshes.",
)
parser.add_argument(
    "--robot_prim",
    type=str,
    default="/World/Robot",
    help="Prim path where the imported robot articulation is spawned.",
)
parser.add_argument("--left_ee_body", type=str, default="LinearclampinggripperJZ02_Link")
parser.add_argument("--right_ee_body", type=str, default="LinearclampinggripperJZ01_Link")
parser.add_argument("--goal_hold_steps", type=int, default=180, help="Simulation steps to hold each IK target.")
parser.add_argument("--no_task_scene", action="store_true", help="Spawn only ground/light/robot; skip table, bucket, and cubes.")
parser.add_argument("--print_only", action="store_true", help="Only print names and validation results; do not run IK.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.sim.utils import get_current_stage
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms
from pxr import Usd, UsdGeom


# This is the same movable-joint order assumed by the X-VLA custom_handler.py FK adapter.
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
LEFT_ARM_JOINTS = [f"ArmL0{i}_Joint" for i in range(2, 9)]
RIGHT_ARM_JOINTS = [f"ArmR0{i}_Joint" for i in range(2, 9)]
LEFT_GRIPPER_JOINTS = ["JawBlock03_Joint", "JawBlock04_Joint"]
RIGHT_GRIPPER_JOINTS = ["JawBlock01_Joint", "JawBlock02_Joint"]


@dataclass(frozen=True)
class ArmIkContext:
    name: str
    entity_cfg: SceneEntityCfg
    controller: DifferentialIKController
    jacobian_body_index: int
    goals_b: torch.Tensor
    marker_current: VisualizationMarkers
    marker_goal: VisualizationMarkers


def _abs_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _make_robot_cfg() -> ArticulationCfg:
    """Import BRX from URDF and wrap it as an Isaac Lab articulation."""
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
            # Keep fixed-joint child links so EE body names from training FK stay available.
            merge_fixed_joints=False,
            self_collision=False,
            collision_from_visuals=False,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=800.0, damping=40.0),
                target_type="position",
                drive_type="force",
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
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
                effort_limit_sim=300.0,
                velocity_limit_sim=20.0,
                stiffness=800.0,
                damping=40.0,
            )
        },
    )


def _make_material(color: tuple[float, float, float], roughness: float = 0.7) -> sim_utils.PreviewSurfaceCfg:
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=roughness)


def _spawn_static_cuboid(
    prim_path: str,
    size: tuple[float, float, float],
    translation: tuple[float, float, float],
    color: tuple[float, float, float],
) -> None:
    cfg = sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=_make_material(color),
    )
    cfg.func(prim_path, cfg, translation=translation)


def _spawn_rigid_cube(
    prim_path: str,
    size: float,
    translation: tuple[float, float, float],
    color: tuple[float, float, float],
) -> None:
    cfg = sim_utils.CuboidCfg(
        size=(size, size, size),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            max_depenetration_velocity=1.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6, restitution=0.0),
        visual_material=_make_material(color),
    )
    cfg.func(prim_path, cfg, translation=translation)



def _spawn_floor_visuals() -> None:
    """Add a clean lab-style tile overlay above the physics ground plane."""
    sim_utils.create_prim("/World/FloorVisuals", "Xform")
    floor_cfg = sim_utils.CuboidCfg(
        size=(4.0, 4.0, 0.004),
        visual_material=_make_material((0.72, 0.76, 0.76), roughness=0.9),
    )
    floor_cfg.func("/World/FloorVisuals/Base", floor_cfg, translation=(0.35, 0.0, -0.003))

    line_color = (0.50, 0.55, 0.55)
    line_width = 0.006
    line_height = 0.002
    extent = 4.0
    spacing = 0.25
    for idx in range(-8, 9):
        offset = idx * spacing
        line_x_cfg = sim_utils.CuboidCfg(
            size=(line_width, extent, line_height),
            visual_material=_make_material(line_color, roughness=0.95),
        )
        line_x_cfg.func(f"/World/FloorVisuals/GridX_{idx + 8:02d}", line_x_cfg, translation=(0.35 + offset, 0.0, 0.001))

        line_y_cfg = sim_utils.CuboidCfg(
            size=(extent, line_width, line_height),
            visual_material=_make_material(line_color, roughness=0.95),
        )
        line_y_cfg.func(f"/World/FloorVisuals/GridY_{idx + 8:02d}", line_y_cfg, translation=(0.35, offset, 0.001))
def _spawn_bucket(prefix: str, center: tuple[float, float, float]) -> None:
    """Spawn a simple open-top bucket using five static cuboids."""
    x, y, table_z = center
    wall_thickness = 0.018
    outer = 0.20
    height = 0.16
    bottom_thickness = 0.018
    base_z = table_z + bottom_thickness * 0.5
    wall_z = table_z + bottom_thickness + height * 0.5
    bucket_color = (0.95, 0.72, 0.18)

    _spawn_static_cuboid(f"{prefix}/Bottom", (outer, outer, bottom_thickness), (x, y, base_z), bucket_color)
    _spawn_static_cuboid(f"{prefix}/WallPosX", (wall_thickness, outer, height), (x + outer * 0.5, y, wall_z), bucket_color)
    _spawn_static_cuboid(f"{prefix}/WallNegX", (wall_thickness, outer, height), (x - outer * 0.5, y, wall_z), bucket_color)
    _spawn_static_cuboid(f"{prefix}/WallPosY", (outer, wall_thickness, height), (x, y + outer * 0.5, wall_z), bucket_color)
    _spawn_static_cuboid(f"{prefix}/WallNegY", (outer, wall_thickness, height), (x, y - outer * 0.5, wall_z), bucket_color)


def _spawn_pick_place_scene() -> None:
    """Spawn a simple table-top scene: table, bucket, and two graspable cubes."""
    sim_utils.create_prim("/World/TaskScene", "Xform")

    table_center_x = 0.72
    table_center_y = 0.0
    table_top_z = 0.74
    tabletop_thickness = 0.055
    tabletop_center_z = table_top_z - tabletop_thickness * 0.5

    _spawn_static_cuboid(
        "/World/TaskScene/TableTop",
        (0.78, 0.72, tabletop_thickness),
        (table_center_x, table_center_y, tabletop_center_z),
        (0.48, 0.42, 0.34),
    )
    for name, dx, dy in [
        ("LegFL", 0.31, 0.27),
        ("LegFR", 0.31, -0.27),
        ("LegBL", -0.31, 0.27),
        ("LegBR", -0.31, -0.27),
    ]:
        _spawn_static_cuboid(
            f"/World/TaskScene/{name}",
            (0.045, 0.045, table_top_z),
            (table_center_x + dx, table_center_y + dy, table_top_z * 0.5),
            (0.34, 0.30, 0.25),
        )

    _spawn_bucket("/World/TaskScene/Bucket", (0.82, 0.0, table_top_z))
    cube_size = 0.06
    cube_z = table_top_z + cube_size * 0.5 + 0.003
    _spawn_rigid_cube("/World/TaskScene/BlockRed", cube_size, (0.56, 0.16, cube_z), (0.9, 0.12, 0.08))
    _spawn_rigid_cube("/World/TaskScene/BlockBlue", cube_size, (0.56, -0.16, cube_z), (0.08, 0.22, 0.9))

    print("[BRX] Spawned task scene: table, open-top bucket, and two rigid cubes.")


def _spawn_minimal_scene() -> None:
    """Create the scene objects needed for IK and task-scene checks."""
    ground_cfg = sim_utils.GroundPlaneCfg(
        color=(0.72, 0.76, 0.76),
        size=(8.0, 8.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9, restitution=0.0),
    )
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    _spawn_floor_visuals()

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
    light_cfg.func("/World/Light", light_cfg)

    if not args_cli.no_task_scene:
        _spawn_pick_place_scene()


def _converted_usd_path() -> str:
    urdf_path = _abs_path(args_cli.urdf_path)
    usd_dir = _abs_path(args_cli.usd_dir) if args_cli.usd_dir else os.path.join(os.path.dirname(urdf_path), "isaaclab_converted")
    return os.path.join(usd_dir, f"{os.path.splitext(os.path.basename(urdf_path))[0]}_imported.usd")


def _print_stage_visual_summary() -> None:
    """Print whether visual mesh prims exist under the spawned robot prim."""
    stage = get_current_stage()
    robot_prim = stage.GetPrimAtPath(args_cli.robot_prim)
    if not robot_prim.IsValid():
        print(f"[BRX] robot prim is not valid on stage: {args_cli.robot_prim}")
        return

    mesh_paths = []
    imageable_paths = []
    for prim in Usd.PrimRange(robot_prim):
        if prim.IsA(UsdGeom.Mesh):
            mesh_paths.append(str(prim.GetPath()))
        if prim.IsA(UsdGeom.Imageable):
            imageable_paths.append(str(prim.GetPath()))

    print("\n[BRX] Stage visual summary")
    print("[BRX] converted USD path:", _converted_usd_path())
    print("[BRX] mesh prim count under robot:", len(mesh_paths))
    print("[BRX] imageable prim count under robot:", len(imageable_paths))
    for path in mesh_paths[:20]:
        print("  mesh:", path)
    if len(mesh_paths) > 20:
        print(f"  ... {len(mesh_paths) - 20} more mesh prims")
    if not mesh_paths:
        print("[BRX] No Mesh prims were found under the robot. The articulation can exist while visuals are missing.")
        print("[BRX] Check that the URDF folder contains meshes/*.STL and rerun with --force_usd_conversion --no_instanceable.")


def _print_body_pose_summary(robot: Articulation) -> None:
    body_pos_w = robot.data.body_state_w[0, :, 0:3]
    mins = torch.min(body_pos_w, dim=0).values
    maxs = torch.max(body_pos_w, dim=0).values
    root = robot.data.root_pose_w[0]

    print("\n[BRX] Body pose summary")
    print("[BRX] root xyz:", root[0:3].detach().cpu().tolist())
    print("[BRX] body xyz min:", mins.detach().cpu().tolist())
    print("[BRX] body xyz max:", maxs.detach().cpu().tolist())
    for idx, name in enumerate(robot.body_names[:20]):
        print(f"  body_pos[{idx:02d}] {name}: {body_pos_w[idx].detach().cpu().tolist()}")
    if len(robot.body_names) > 20:
        print(f"  ... {len(robot.body_names) - 20} more bodies")


def _set_camera_to_robot(sim: SimulationContext, robot: Articulation) -> None:
    body_pos_w = robot.data.body_state_w[0, :, 0:3]
    mins = torch.min(body_pos_w, dim=0).values
    maxs = torch.max(body_pos_w, dim=0).values
    center = 0.5 * (mins + maxs)
    span = torch.clamp(maxs - mins, min=0.25)
    distance = float(torch.max(span).detach().cpu()) * 2.5
    distance = max(distance, 1.5)
    eye = center + torch.tensor([distance, -distance, distance * 0.7], device=body_pos_w.device, dtype=body_pos_w.dtype)
    sim.set_camera_view(eye.detach().cpu().tolist(), center.detach().cpu().tolist())

def _print_validation(robot: Articulation) -> None:
    print("\n[BRX] URDF import input")
    print("[BRX] urdf_path:", _abs_path(args_cli.urdf_path))
    print("[BRX] robot prim:", args_cli.robot_prim)

    print("\n[BRX] Resolved robot names")
    print("[BRX] joint count:", len(robot.joint_names))
    for idx, name in enumerate(robot.joint_names):
        print(f"  joint[{idx:02d}] {name}")
    print("[BRX] body count:", len(robot.body_names))
    for idx, name in enumerate(robot.body_names):
        print(f"  body[{idx:02d}] {name}")

    joint_set = set(robot.joint_names)
    body_set = set(robot.body_names)
    missing_expected = [name for name in EXPECTED_MOVABLE_JOINTS if name not in joint_set]
    missing_left = [name for name in LEFT_ARM_JOINTS + LEFT_GRIPPER_JOINTS if name not in joint_set]
    missing_right = [name for name in RIGHT_ARM_JOINTS + RIGHT_GRIPPER_JOINTS if name not in joint_set]
    missing_bodies = [name for name in [args_cli.left_ee_body, args_cli.right_ee_body] if name not in body_set]

    print("\n[BRX] X-VLA FK assumption check")
    print("[BRX] expected 23 movable joints present:", not missing_expected)
    if missing_expected:
        print("[BRX] missing expected movable joints:", missing_expected)
    print("[BRX] left arm/gripper names present:", not missing_left)
    if missing_left:
        print("[BRX] missing left names:", missing_left)
    print("[BRX] right arm/gripper names present:", not missing_right)
    if missing_right:
        print("[BRX] missing right names:", missing_right)
    print("[BRX] EE body names present:", not missing_bodies)
    if missing_bodies:
        print("[BRX] missing EE bodies:", missing_bodies)

    ordered_overlap = [name for name in EXPECTED_MOVABLE_JOINTS if name in joint_set]
    print("[BRX] expected movable joint order used by training FK:")
    for idx, name in enumerate(ordered_overlap):
        print(f"  fk_q[{idx:02d}] {name} -> isaac_joint_index={robot.joint_names.index(name)}")


def _make_marker(path: str) -> VisualizationMarkers:
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
    return VisualizationMarkers(marker_cfg.replace(prim_path=path))


def _resolve_arm(
    sim: SimulationContext,
    robot: Articulation,
    entity_name: str,
    joint_names: list[str],
    ee_body_name: str,
    goal_offsets: list[list[float]],
) -> ArmIkContext:
    entity_cfg = SceneEntityCfg("robot", joint_names=joint_names, body_names=[ee_body_name])
    scene_map = {"robot": robot}
    entity_cfg.resolve(scene_map)

    controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    controller = DifferentialIKController(controller_cfg, num_envs=1, device=sim.device)

    if robot.is_fixed_base:
        jacobian_body_index = entity_cfg.body_ids[0] - 1
    else:
        jacobian_body_index = entity_cfg.body_ids[0]

    root_pose_w = robot.data.root_pose_w
    ee_pose_w = robot.data.body_pose_w[:, entity_cfg.body_ids[0]]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )

    goals = []
    for offset in goal_offsets:
        goal_pos = ee_pos_b[0] + torch.tensor(offset, device=sim.device, dtype=ee_pos_b.dtype)
        goals.append(torch.cat([goal_pos, ee_quat_b[0]], dim=0))
    goals_b = torch.stack(goals, dim=0)

    return ArmIkContext(
        name=entity_name,
        entity_cfg=entity_cfg,
        controller=controller,
        jacobian_body_index=jacobian_body_index,
        goals_b=goals_b,
        marker_current=_make_marker(f"/Visuals/{entity_name}_ee_current"),
        marker_goal=_make_marker(f"/Visuals/{entity_name}_ee_goal"),
    )


def _compute_arm_command(robot: Articulation, ctx: ArmIkContext) -> torch.Tensor:
    jacobian = robot.root_physx_view.get_jacobians()[:, ctx.jacobian_body_index, :, ctx.entity_cfg.joint_ids]
    ee_pose_w = robot.data.body_pose_w[:, ctx.entity_cfg.body_ids[0]]
    root_pose_w = robot.data.root_pose_w
    joint_pos = robot.data.joint_pos[:, ctx.entity_cfg.joint_ids]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    return ctx.controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)


def _apply_gripper_targets(robot: Articulation, open_amount: float) -> None:
    """Command both grippers symmetrically. open_amount is clamped to [0, 1]."""
    open_amount = max(0.0, min(1.0, float(open_amount)))
    jaw = 0.041 * open_amount
    targets = torch.tensor([[jaw, -jaw, jaw, -jaw]], device=robot.device)
    names = RIGHT_GRIPPER_JOINTS + LEFT_GRIPPER_JOINTS
    joint_ids = [robot.joint_names.index(name) for name in names if name in robot.joint_names]
    if len(joint_ids) == 4:
        robot.set_joint_position_target(targets, joint_ids=joint_ids)


def run_simulator(sim: SimulationContext, robot: Articulation) -> None:
    sim_dt = sim.get_physics_dt()

    # Step once so articulation buffers and body poses are initialized.
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim_dt)

    _print_validation(robot)
    _print_stage_visual_summary()
    _print_body_pose_summary(robot)
    _set_camera_to_robot(sim, robot)
    if args_cli.print_only:
        return

    left_ctx = _resolve_arm(
        sim,
        robot,
        "left",
        LEFT_ARM_JOINTS,
        args_cli.left_ee_body,
        goal_offsets=[[0.04, 0.00, 0.00], [0.04, 0.04, 0.00], [0.02, 0.02, 0.04]],
    )
    right_ctx = _resolve_arm(
        sim,
        robot,
        "right",
        RIGHT_ARM_JOINTS,
        args_cli.right_ee_body,
        goal_offsets=[[0.04, 0.00, 0.00], [0.04, -0.04, 0.00], [0.02, -0.02, 0.04]],
    )

    count = 0
    while simulation_app.is_running():
        goal_idx = (count // max(1, args_cli.goal_hold_steps)) % left_ctx.goals_b.shape[0]

        if count % max(1, args_cli.goal_hold_steps) == 0:
            left_ctx.controller.reset()
            right_ctx.controller.reset()
            left_ctx.controller.set_command(left_ctx.goals_b[goal_idx : goal_idx + 1])
            right_ctx.controller.set_command(right_ctx.goals_b[goal_idx : goal_idx + 1])
            print(f"[BRX] Switching IK goal index: {goal_idx}")

        left_joint_cmd = _compute_arm_command(robot, left_ctx)
        right_joint_cmd = _compute_arm_command(robot, right_ctx)
        robot.set_joint_position_target(left_joint_cmd, joint_ids=left_ctx.entity_cfg.joint_ids)
        robot.set_joint_position_target(right_joint_cmd, joint_ids=right_ctx.entity_cfg.joint_ids)
        _apply_gripper_targets(robot, open_amount=0.5 + 0.5 * ((goal_idx % 2) == 0))

        robot.write_data_to_sim()
        sim.step()
        count += 1
        robot.update(sim_dt)

        left_pose_w = robot.data.body_state_w[:, left_ctx.entity_cfg.body_ids[0], 0:7]
        right_pose_w = robot.data.body_state_w[:, right_ctx.entity_cfg.body_ids[0], 0:7]
        root_pose_w = robot.data.root_pose_w
        left_goal_pos_w, left_goal_quat_w = combine_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], left_ctx.goals_b[goal_idx : goal_idx + 1, 0:3], left_ctx.goals_b[goal_idx : goal_idx + 1, 3:7]
        )
        right_goal_pos_w, right_goal_quat_w = combine_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], right_ctx.goals_b[goal_idx : goal_idx + 1, 0:3], right_ctx.goals_b[goal_idx : goal_idx + 1, 3:7]
        )

        left_ctx.marker_current.visualize(left_pose_w[:, 0:3], left_pose_w[:, 3:7])
        right_ctx.marker_current.visualize(right_pose_w[:, 0:3], right_pose_w[:, 3:7])
        left_ctx.marker_goal.visualize(left_goal_pos_w, left_goal_quat_w)
        right_ctx.marker_goal.visualize(right_goal_pos_w, right_goal_quat_w)


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, -2.5, 2.0], [0.0, 0.0, 0.8])

    _spawn_minimal_scene()
    robot = Articulation(cfg=_make_robot_cfg())

    sim.reset()
    print("[BRX] Setup complete.")
    run_simulator(sim, robot)


if __name__ == "__main__":
    main()
    simulation_app.close()
