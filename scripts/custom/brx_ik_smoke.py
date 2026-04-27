# Copyright (c) 2026.
# SPDX-License-Identifier: BSD-3-Clause

"""
BRX dual-arm Differential IK smoke test for Isaac Lab.

This script loads a full scene USD that already contains the BRX robot articulation,
prints the resolved joint/body names, checks the names used by the X-VLA FK pipeline,
and drives left/right end-effector bodies through small absolute pose targets.

Usage example:

    ./isaaclab.sh -p scripts/custom/brx_ik_smoke.py \
        --scene_usd /path/to/scene.usd \
        --robot_prim /World/Robot

The first stage intentionally does not call X-VLA. It verifies that the USD,
articulation, body names, joint names, Jacobians, and Differential IK control path
are usable before policy integration.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="BRX dual-arm Differential IK smoke test.")
parser.add_argument("--scene_usd", type=str, required=True, help="Path to the complete scene USD.")
parser.add_argument(
    "--robot_prim",
    type=str,
    default="/World/Robot",
    help="Prim path of the robot articulation inside the loaded scene USD.",
)
parser.add_argument("--left_ee_body", type=str, default="LinearclampinggripperJZ02_Link")
parser.add_argument("--right_ee_body", type=str, default="LinearclampinggripperJZ01_Link")
parser.add_argument("--goal_hold_steps", type=int, default=180, help="Simulation steps to hold each IK target.")
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
from isaaclab.utils.math import subtract_frame_transforms


# This order is the assumption used by the X-VLA custom_handler.py FK adapter.
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


def _make_robot_cfg() -> ArticulationCfg:
    """Wrap an existing USD robot articulation at args_cli.robot_prim."""
    return ArticulationCfg(
        prim_path=args_cli.robot_prim,
        spawn=None,
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


def _load_scene_usd() -> None:
    """Load the full scene USD into /World."""
    cfg = sim_utils.UsdFileCfg(usd_path=args_cli.scene_usd)
    cfg.func("/World", cfg)


def _print_validation(robot: Articulation) -> None:
    print("\n[BRX] Resolved robot names")
    print("[BRX] robot prim:", args_cli.robot_prim)
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
    # Use a tiny shim object with the same interface SceneEntityCfg.resolve expects.
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
        left_goal_pos_w = left_ctx.goals_b[goal_idx : goal_idx + 1, 0:3] + root_pose_w[:, 0:3]
        right_goal_pos_w = right_ctx.goals_b[goal_idx : goal_idx + 1, 0:3] + root_pose_w[:, 0:3]

        left_ctx.marker_current.visualize(left_pose_w[:, 0:3], left_pose_w[:, 3:7])
        right_ctx.marker_current.visualize(right_pose_w[:, 0:3], right_pose_w[:, 3:7])
        left_ctx.marker_goal.visualize(left_goal_pos_w, left_ctx.goals_b[goal_idx : goal_idx + 1, 3:7])
        right_ctx.marker_goal.visualize(right_goal_pos_w, right_ctx.goals_b[goal_idx : goal_idx + 1, 3:7])


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, -2.5, 2.0], [0.0, 0.0, 0.8])

    _load_scene_usd()
    robot = Articulation(cfg=_make_robot_cfg())

    sim.reset()
    print("[BRX] Setup complete.")
    run_simulator(sim, robot)


if __name__ == "__main__":
    main()
    simulation_app.close()
