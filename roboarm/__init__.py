"""Public package interface for roboarm."""

from .kinematics import (
    L1,
    L2,
    L3,
    LINK_LENGTHS,
    forward_kinematics,
    inverse_kinematics_2link,
    inverse_kinematics_numeric,
    jacobian,
)
from .trajectory import cartesian_path, plan_trajectory
from .workspace import check_collision, reachable, sample_workspace

__all__ = [
    "L1",
    "L2",
    "L3",
    "LINK_LENGTHS",
    "forward_kinematics",
    "inverse_kinematics_2link",
    "inverse_kinematics_numeric",
    "jacobian",
    "plan_trajectory",
    "cartesian_path",
    "sample_workspace",
    "check_collision",
    "reachable",
]

__version__ = "0.1.0"
