"""Workspace analysis: Monte-Carlo reachability and collision checks.

The reachable set of the end-effector is the annulus (in the plane)
bounded by the fully stretched arm, radius L1 + L2 + L3 = 1.2 m.  A point
is reachable by the *wrist* (end of link 2) iff it lies in the annulus
|L1 - L2| <= r <= L1 + L2; the end-effector can additionally reach a
little further/closer thanks to link 3, so ``reachable`` here uses the
conservative 2-link test for the wrist plus the full-stretch bound, and
`sample_workspace` samples the true 3-link reachable set directly.
"""

import math

import numpy as np

from .kinematics import L1, L2, L3, LINK_LENGTHS, forward_kinematics


def sample_workspace(n, seed=None):
    """Monte-Carlo sample of reachable end-effector positions.

    Draws `n` random joint configurations uniformly in [-pi, pi]^3 and
    returns the corresponding end-effector (x, y) positions.

    Parameters
    ----------
    n : int
        Number of samples.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    points : ndarray of shape (n, 2)
        Sampled end-effector positions.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    rng = np.random.default_rng(seed)
    thetas = rng.uniform(-math.pi, math.pi, size=(n, 3))
    points = np.empty((n, 2))
    for i, theta in enumerate(thetas):
        _, (x, y, _phi) = forward_kinematics(theta)
        points[i] = (x, y)
    return points


def _segment_circle_distance(p1, p2, c):
    """Shortest distance between segment p1->p2 and point c.

    Standard closest-point-on-segment test: project c onto the line,
    clamp the parameter to [0, 1], measure the distance.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    c = np.asarray(c, dtype=float)
    d = p2 - p1
    length_sq = float(d @ d)
    if length_sq == 0.0:
        return float(np.linalg.norm(c - p1))
    t = float((c - p1) @ d) / length_sq
    t = max(0.0, min(1.0, t))  # clamp to the segment
    closest = p1 + t * d
    return float(np.linalg.norm(c - closest))


def check_collision(joint_positions, obstacle):
    """Check whether any arm link intersects a circular obstacle.

    Parameters
    ----------
    joint_positions : list of (x, y)
        Joint positions as returned by ``forward_kinematics`` (base first).
    obstacle : (cx, cy, r)
        Circle centre and radius.

    Returns
    -------
    bool
        True if any link segment comes within radius r of the obstacle
        centre (touching counts as a collision).
    """
    cx, cy, r = obstacle
    centre = (cx, cy)
    for p1, p2 in zip(joint_positions[:-1], joint_positions[1:]):
        if _segment_circle_distance(p1, p2, centre) <= r:
            return True
    return False


def reachable(x, y):
    """Boolean 2-link reachability test for a wrist target (x, y).

    A point is reachable by the end of link 2 iff its distance from the
    base lies in the annulus |L1 - L2| <= r <= L1 + L2.
    """
    r = math.hypot(x, y)
    return abs(L1 - L2) - 1e-12 <= r <= L1 + L2 + 1e-12


def max_reach():
    """Maximum end-effector reach (fully stretched arm), in metres."""
    return sum(LINK_LENGTHS)
