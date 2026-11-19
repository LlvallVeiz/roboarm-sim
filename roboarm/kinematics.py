"""Forward / inverse kinematics for a 3-DOF planar robot arm.

The arm is a planar serial chain with three revolute joints rotating about
axes perpendicular to the xy-plane (all joint axes parallel to z).  Each
link i has length L_i and joint angle theta_i measured relative to the
previous link, so the absolute orientation of link i is

    phi_i = theta_1 + theta_2 + ... + theta_i

and the position of joint i+1 (the far end of link i) is

    p_{i+1} = p_i + L_i * (cos phi_i, sin phi_i),   p_0 = (0, 0).

This is the planar special case of the Denavit-Hartenberg convention with
a_i = L_i, alpha_i = 0, d_i = 0, theta_i as above (the classic "RRR planar
arm" DH table).  The end-effector pose is (x, y, phi) with
phi = theta_1 + theta_2 + theta_3.
"""

import math

import numpy as np

# Link lengths (m), fixed by design.
L1 = 0.5
L2 = 0.4
L3 = 0.3

LINK_LENGTHS = (L1, L2, L3)
N_JOINTS = 3


def _as_array(theta):
    """Validate a joint vector and return it as a float ndarray of length 3."""
    q = np.asarray(theta, dtype=float).ravel()
    if q.shape != (N_JOINTS,):
        raise ValueError(
            f"expected 3 joint angles, got shape {q.shape} from {theta!r}"
        )
    return q


def forward_kinematics(theta):
    """Compute joint positions and the end-effector pose.

    Parameters
    ----------
    theta : array-like of 3 floats
        Joint angles [theta_1, theta_2, theta_3] in radians, each relative
        to the previous link.

    Returns
    -------
    positions : list of (x, y) tuples
        Positions of the base (0, 0) and the end of each link, i.e. four
        points: base, joint 2, joint 3, end-effector.
    pose : (x, y, phi)
        End-effector position and absolute orientation phi (radians).
    """
    q = _as_array(theta)
    positions = [(0.0, 0.0)]
    x, y = 0.0, 0.0
    phi = 0.0
    for theta_i, L_i in zip(q, LINK_LENGTHS):
        phi += theta_i
        x += L_i * math.cos(phi)
        y += L_i * math.sin(phi)
        positions.append((x, y))
    return positions, (x, y, phi)


def jacobian(theta):
    """Geometric Jacobian of the end-effector pose w.r.t. joint angles.

    For a planar arm the pose is (x, y, phi); the Jacobian is 3x3 where
    row 0/1 are the partial derivatives of the end-effector position and
    row 2 is d(phi)/d(theta_j) = 1 for every joint.  The position part
    follows from differentiating the forward-kinematics sum: joint j
    rotates every link i >= j, so

        dx/d(theta_j) = -sum_{i>=j} L_i * sin(phi_i)
        dy/d(theta_j) =  sum_{i>=j} L_i * cos(phi_i)
    """
    q = _as_array(theta)
    # Absolute link orientations phi_i.
    phis = np.cumsum(q)
    J = np.zeros((3, 3))
    for j in range(N_JOINTS):
        # Links j..2 (0-based) move when joint j rotates.
        s = sum(
            LINK_LENGTHS[i] * math.sin(phis[i]) for i in range(j, N_JOINTS)
        )
        c = sum(
            LINK_LENGTHS[i] * math.cos(phis[i]) for i in range(j, N_JOINTS)
        )
        J[0, j] = -s
        J[1, j] = c
        J[2, j] = 1.0
    return J


def inverse_kinematics_2link(x, y, elbow_up=True):
    """Analytic IK for the first two links (position of the wrist).

    Solves for (theta_1, theta_2) that place the end of link 2 at (x, y)
    using the law of cosines.  Two solutions exist (elbow up / down); the
    `elbow_up` flag selects between them (elbow up = theta_2 > 0 in our
    convention, i.e. the elbow bends counter-clockwise).

    Parameters
    ----------
    x, y : float
        Target position of the end of link 2 (the "wrist").
    elbow_up : bool
        If True pick the elbow-up branch, else elbow-down.

    Returns
    -------
    (theta_1, theta_2) : tuple of floats, radians.

    Raises
    ------
    ValueError
        If the target is outside the annulus |L1 - L2| <= r <= L1 + L2.
    """
    r = math.hypot(x, y)
    if r > L1 + L2 + 1e-12:
        raise ValueError(
            f"target ({x:.4f}, {y:.4f}) unreachable: distance {r:.4f} "
            f"> L1 + L2 = {L1 + L2:.4f}"
        )
    if r < abs(L1 - L2) - 1e-12:
        raise ValueError(
            f"target ({x:.4f}, {y:.4f}) unreachable: distance {r:.4f} "
            f"< |L1 - L2| = {abs(L1 - L2):.4f}"
        )
    # Law of cosines: cos(theta_2) = (r^2 - L1^2 - L2^2) / (2 L1 L2).
    cos_t2 = (r * r - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    cos_t2 = max(-1.0, min(1.0, cos_t2))  # guard against float round-off
    theta_2 = math.acos(cos_t2)
    if not elbow_up:
        theta_2 = -theta_2
    # theta_1 = bearing of target - angle between link 1 and the target ray.
    # atan2(L2 sin t2, L1 + L2 cos t2) is that offset (sine rule form).
    theta_1 = math.atan2(y, x) - math.atan2(
        L2 * math.sin(theta_2), L1 + L2 * math.cos(theta_2)
    )
    return theta_1, theta_2


def _pose_error(target, q):
    """Pose error (dx, dy, dphi) with dphi wrapped to [-pi, pi]."""
    _, (x, y, phi) = forward_kinematics(q)
    err = target - np.array([x, y, phi])
    err[2] = (err[2] + math.pi) % (2.0 * math.pi) - math.pi
    return err


def inverse_kinematics_numeric(target, theta0, tol=1e-6, max_iter=200,
                               damping=1e-6, max_step=0.5):
    """Iterative IK for all 3 joints via the Jacobian pseudo-inverse.

    Solves for theta such that the end-effector pose matches `target`.
    Uses damped least-squares (Levenberg-Marquardt style) Newton steps:

        dtheta = J^T (J J^T + lambda^2 I)^-1 * error

    which is numerically the Moore-Penrose pseudo-inverse J^+ when the
    damping is tiny, but stays well behaved near singular configurations.

    The pose error is (dx, dy, dphi) with dphi wrapped to [-pi, pi] so the
    orientation error is always the shortest angular difference.

    Plain Newton steps can overshoot badly near singularities (the
    Jacobian inverse blows up), so each step is capped at `max_step`
    radians and accepted only if a backtracking line search finds a
    step length that actually reduces the pose error; otherwise the
    solver stops and reports non-convergence instead of diverging.

    Parameters
    ----------
    target : (x, y, phi)
        Desired end-effector pose, phi in radians.
    theta0 : array-like of 3 floats
        Initial joint guess (radians).
    tol : float
        Convergence tolerance on the Euclidean pose-error norm.
    max_iter : int
        Maximum Newton iterations.
    damping : float
        Damping factor lambda for the least-squares step.
    max_step : float
        Maximum joint-step size (radians) per iteration.

    Returns
    -------
    theta : ndarray of 3 floats
        Joint angles reaching `target` (within `tol`).

    Raises
    ------
    ValueError
        If no convergence within `max_iter` iterations.
    """
    q = _as_array(theta0)
    target = np.asarray(target, dtype=float).ravel()
    if target.shape != (3,):
        raise ValueError(f"expected target pose (x, y, phi), got {target!r}")

    err = _pose_error(target, q)
    for _ in range(max_iter):
        if np.linalg.norm(err) < tol:
            return q
        J = jacobian(q)
        # Damped pseudo-inverse step: J^T (J J^T + lambda^2 I)^-1 err.
        A = J @ J.T + (damping ** 2) * np.eye(3)
        step = J.T @ np.linalg.solve(A, err)
        # Cap the step size so near-singular Jacobians cannot throw the
        # solver across the workspace in one iteration.
        step_norm = np.linalg.norm(step)
        if step_norm > max_step:
            step *= max_step / step_norm
        # Backtracking line search: only accept a step that reduces the
        # pose error (halve up to 20 times, then give up on this step).
        accepted = False
        alpha = 1.0
        for _halving in range(21):
            q_new = q + alpha * step
            err_new = _pose_error(target, q_new)
            if np.linalg.norm(err_new) < np.linalg.norm(err):
                q, err = q_new, err_new
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break

    if np.linalg.norm(err) < tol:
        return q
    raise ValueError(
        f"numeric IK did not converge in {max_iter} iterations "
        f"(final pose error {np.linalg.norm(err):.2e})"
    )
