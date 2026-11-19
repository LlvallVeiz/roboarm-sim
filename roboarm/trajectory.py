"""Joint-space trajectory planning and Cartesian waypoint following.

The planner uses a *cubic* (third-order polynomial) profile per joint,
which is the simplest profile with zero velocity at both ends.  For a
single joint moving from q0 to q1 over duration T, the normalised time
s = t / T in [0, 1] gives

    q(s) = q0 + (q1 - q0) * (3 s^2 - 2 s^3)
    q'(s) = (q1 - q0) * (6 s - 6 s^2) / T

so q(0) = q0, q(1) = q1 exactly, and q'(0) = q'(1) = 0 (smooth start and
stop, bell-shaped velocity, peak velocity 1.5 * (q1 - q0) / T at s = 0.5).
"""

import math

import numpy as np

from .kinematics import forward_kinematics, inverse_kinematics_numeric


def _cubic_profile(q0, q1, s):
    """Cubic blend factor and its time derivative at normalised time s."""
    # Position blend: 3s^2 - 2s^3 ; velocity blend (w.r.t. s): 6s - 6s^2.
    pos = 3.0 * s * s - 2.0 * s * s * s
    vel = 6.0 * s - 6.0 * s * s
    return pos, vel


def plan_trajectory(q_start, q_goal, duration, steps):
    """Plan a smooth joint-space trajectory from q_start to q_goal.

    Parameters
    ----------
    q_start, q_goal : array-like of 3 floats
        Start and goal joint configurations (radians).
    duration : float
        Total motion time T in seconds (> 0).
    steps : int
        Number of samples; the returned list has ``steps + 1`` entries
        (t = 0 .. T inclusive).

    Returns
    -------
    trajectory : list of dicts
        Each entry has keys ``t`` (time), ``q`` (joint angles, ndarray),
        and ``qd`` (joint velocities, ndarray).  The first entry is
        exactly ``q_start`` at t = 0 with zero velocity and the last is
        exactly ``q_goal`` at t = T with zero velocity.
    """
    q0 = np.asarray(q_start, dtype=float).ravel()
    q1 = np.asarray(q_goal, dtype=float).ravel()
    if q0.shape != (3,) or q1.shape != (3,):
        raise ValueError("q_start and q_goal must each have 3 joint angles")
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    dq = q1 - q0
    trajectory = []
    for i in range(steps + 1):
        s = i / steps  # normalised time in [0, 1]
        blend, vel_blend = _cubic_profile(q0, q1, s)
        q = q0 + dq * blend
        qd = dq * vel_blend / duration
        # Pin the endpoints exactly (avoid float drift in the blend).
        if i == 0:
            q = q0.copy()
            qd = np.zeros(3)
        elif i == steps:
            q = q1.copy()
            qd = np.zeros(3)
        trajectory.append({"t": s * duration, "q": q, "qd": qd})
    return trajectory


def cartesian_path(points, theta0=None, tol=1e-6, max_iter=200):
    """Run numeric IK along a list of Cartesian waypoints.

    Each waypoint is an (x, y, phi) end-effector pose.  The solution for
    one waypoint seeds the next (warm start), which keeps the arm on a
    continuous branch instead of jumping between IK solutions.

    Parameters
    ----------
    points : list of (x, y, phi)
        Waypoints to follow in order.
    theta0 : array-like of 3 floats, optional
        Initial joint guess for the first waypoint (default: all zeros).
    tol, max_iter : passed through to the numeric IK solver.

    Returns
    -------
    list of ndarrays
        Joint configuration for each waypoint, in the same order.

    Raises
    ------
    ValueError
        If any waypoint cannot be reached (IK fails to converge).
    """
    if not points:
        return []
    q = np.zeros(3) if theta0 is None else np.asarray(theta0, dtype=float)
    configs = []
    for target in points:
        q = inverse_kinematics_numeric(target, q, tol=tol, max_iter=max_iter)
        configs.append(q.copy())
    return configs


def trajectory_end_effector_path(trajectory):
    """Map a joint trajectory to end-effector (x, y, phi) poses."""
    poses = []
    for entry in trajectory:
        _, pose = forward_kinematics(entry["q"])
        poses.append(pose)
    return poses
