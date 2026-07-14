"""Canonical BRX042501 joint and camera contracts.

The 23-dimensional order in this module is the public data/control ABI.  Do
not derive it from Isaac Lab's articulation order or from URDF declaration
order: both contain wheel joints and historical scripts in this repository
used several incompatible arm/gripper permutations.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


JOINT_NAMES_23: tuple[str, ...] = (
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
)

JOINT_INDEX_23: dict[str, int] = {name: index for index, name in enumerate(JOINT_NAMES_23)}

LEFT_ARM_JOINTS: tuple[str, ...] = tuple(f"ArmL0{i}_Joint" for i in range(2, 9))
RIGHT_ARM_JOINTS: tuple[str, ...] = tuple(f"ArmR0{i}_Joint" for i in range(2, 9))

# Physical side mapping.  The historical 23-D ABI deliberately places the
# right gripper after the left arm and the left gripper after the right arm.
RIGHT_GRIPPER_JOINTS: tuple[str, ...] = ("JawBlock01_Joint", "JawBlock02_Joint")
LEFT_GRIPPER_JOINTS: tuple[str, ...] = ("JawBlock03_Joint", "JawBlock04_Joint")
RIGHT_GRIPPER_INDICES: tuple[int, int] = tuple(JOINT_INDEX_23[name] for name in RIGHT_GRIPPER_JOINTS)
LEFT_GRIPPER_INDICES: tuple[int, int] = tuple(JOINT_INDEX_23[name] for name in LEFT_GRIPPER_JOINTS)

CAMERA_NAMES: tuple[str, ...] = ("head_left", "head_right", "left_wrist", "right_wrist")
CAMERA_LINK_BY_NAME: dict[str, str] = {
    "head_left": "EyeL_Link",
    "head_right": "EyeR_Link",
    "left_wrist": "HandCam02_Link",
    "right_wrist": "HandCam01_Link",
}
CAMERA_FEATURE_BY_NAME: dict[str, str] = {
    name: f"observation.images.{name}" for name in CAMERA_NAMES
}

LEFT_EE_LINK = "LinearclampinggripperJZ02_Link"
RIGHT_EE_LINK = "LinearclampinggripperJZ01_Link"
GRIPPER_OPEN_M = 0.0
GRIPPER_CLOSED_M = 0.041


def require_vector23(values: Sequence[float], label: str) -> list[float]:
    """Validate and copy a finite 23-D numeric vector."""

    if len(values) != len(JOINT_NAMES_23):
        raise ValueError(f"{label} must contain exactly 23 values, got {len(values)}")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} contains NaN or Inf")
    return result


def resolve_joint_ids(articulation_joint_names: Sequence[str]) -> list[int]:
    """Resolve the public 23-D order into Isaac Lab articulation indices."""

    index = {name: idx for idx, name in enumerate(articulation_joint_names)}
    missing = [name for name in JOINT_NAMES_23 if name not in index]
    if missing:
        raise RuntimeError(f"Robot is missing required BRX 23-D joints: {missing}")
    return [index[name] for name in JOINT_NAMES_23]


def names_for_indices(indices: Iterable[int]) -> list[str]:
    return [JOINT_NAMES_23[int(index)] for index in indices]


@dataclass(frozen=True)
class UrdfContractReport:
    path: Path
    link_count: int
    joint_count: int
    non_fixed_joint_count: int
    head_stereo_baseline_m: float
    camera_parent_by_link: dict[str, str]
    joint_limits: dict[str, tuple[float | None, float | None]]


def inspect_urdf_contract(path: str | Path) -> UrdfContractReport:
    """Parse the URDF and fail early if the BRX control/camera ABI is broken."""

    urdf_path = Path(path).expanduser().resolve()
    root = ET.parse(urdf_path).getroot()
    links = {node.attrib["name"] for node in root.findall("link")}
    joints = root.findall("joint")
    joint_by_name = {node.attrib["name"]: node for node in joints}

    missing_joints = [name for name in JOINT_NAMES_23 if name not in joint_by_name]
    missing_links = [link for link in CAMERA_LINK_BY_NAME.values() if link not in links]
    if missing_joints or missing_links:
        raise ValueError(
            f"URDF contract mismatch: missing_joints={missing_joints}, missing_camera_links={missing_links}"
        )

    child_to_parent: dict[str, str] = {}
    child_to_origin: dict[str, tuple[float, float, float]] = {}
    joint_limits: dict[str, tuple[float | None, float | None]] = {}
    for joint in joints:
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is not None and child is not None:
            child_name = child.attrib["link"]
            child_to_parent[child_name] = parent.attrib["link"]
            origin = joint.find("origin")
            xyz = "0 0 0" if origin is None else origin.attrib.get("xyz", "0 0 0")
            child_to_origin[child_name] = tuple(float(value) for value in xyz.split())
        if joint.attrib["name"] in JOINT_INDEX_23:
            limit = joint.find("limit")
            lower = None if limit is None or "lower" not in limit.attrib else float(limit.attrib["lower"])
            upper = None if limit is None or "upper" not in limit.attrib else float(limit.attrib["upper"])
            joint_limits[joint.attrib["name"]] = (lower, upper)

    expected_parents = {
        "EyeL_Link": "Head03_Link",
        "EyeR_Link": "Head03_Link",
        "HandCam02_Link": LEFT_EE_LINK,
        "HandCam01_Link": RIGHT_EE_LINK,
    }
    bad_parents = {
        child: (child_to_parent.get(child), expected)
        for child, expected in expected_parents.items()
        if child_to_parent.get(child) != expected
    }
    if bad_parents:
        raise ValueError(f"URDF camera parent mismatch: {bad_parents}")

    left = child_to_origin["EyeL_Link"]
    right = child_to_origin["EyeR_Link"]
    baseline = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
    if not 0.045 <= baseline <= 0.080:
        raise ValueError(f"Head stereo baseline is unreasonable: {baseline:.6f} m")

    return UrdfContractReport(
        path=urdf_path,
        link_count=len(links),
        joint_count=len(joints),
        non_fixed_joint_count=sum(joint.attrib.get("type", "fixed") != "fixed" for joint in joints),
        head_stereo_baseline_m=baseline,
        camera_parent_by_link={link: child_to_parent[link] for link in CAMERA_LINK_BY_NAME.values()},
        joint_limits=joint_limits,
    )

