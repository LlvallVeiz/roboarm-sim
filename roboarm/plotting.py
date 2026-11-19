"""Matplotlib visualisation and SVG rendering of the arm.

All matplotlib work uses the non-interactive ``Agg`` backend so the
functions work headless (CI, servers, no display).
"""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .kinematics import LINK_LENGTHS, forward_kinematics
from .workspace import max_reach


def plot_arm(theta, ax=None, obstacle=None, show=True):
    """Draw the arm in a given joint configuration.

    Parameters
    ----------
    theta : array-like of 3 floats
        Joint angles in radians.
    ax : matplotlib Axes, optional
        Axes to draw on (a new figure is created if omitted).
    obstacle : (cx, cy, r), optional
        Circular obstacle to draw.
    show : bool
        If True, set equal aspect and sensible axis limits.

    Returns
    -------
    ax : matplotlib Axes
    """
    positions, (x, y, phi) = forward_kinematics(theta)
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    reach = max_reach()

    if ax is None:
        _fig, ax = plt.subplots(figsize=(6, 6))
    if show:
        ax.set_xlim(-reach - 0.1, reach + 0.1)
        ax.set_ylim(-reach - 0.1, reach + 0.1)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("3-DOF planar robot arm")

    # Reachable boundary (dashed circle) for reference.
    circle = plt.Circle((0, 0), reach, fill=False, linestyle="--",
                        color="gray", alpha=0.5, label="max reach")
    ax.add_patch(circle)

    if obstacle is not None:
        cx, cy, r = obstacle
        ax.add_patch(plt.Circle((cx, cy), r, color="red", alpha=0.3,
                                label="obstacle"))

    ax.plot(xs, ys, "-o", color="#1f77b4", linewidth=3, markersize=9,
            markerfacecolor="white", markeredgewidth=2)
    # End-effector orientation arrow.
    ax.annotate("", xy=(x + 0.12 * math.cos(phi), y + 0.12 * math.sin(phi)),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color="green", lw=2))
    ax.plot(0, 0, "ks", markersize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_workspace(points, path=None, out=None, obstacle=None):
    """Scatter the sampled workspace, optionally with a Cartesian path.

    Parameters
    ----------
    points : ndarray of shape (n, 2)
        Sampled end-effector positions (see ``workspace.sample_workspace``).
    path : list of (x, y, phi), optional
        End-effector poses to overlay as a connected path.
    out : str, optional
        If given, save the figure as a PNG to this path.
    obstacle : (cx, cy, r), optional
        Circular obstacle to draw.

    Returns
    -------
    fig : matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(points[:, 0], points[:, 1], s=2, alpha=0.4,
               color="#1f77b4", label="reachable samples")
    reach = max_reach()
    ax.add_patch(plt.Circle((0, 0), reach, fill=False, linestyle="--",
                            color="gray", alpha=0.7,
                            label=f"max reach {reach:.1f} m"))
    if obstacle is not None:
        cx, cy, r = obstacle
        ax.add_patch(plt.Circle((cx, cy), r, color="red", alpha=0.3,
                                label="obstacle"))
    if path:
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        ax.plot(px, py, "r-o", linewidth=2, markersize=5,
                label="end-effector path")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Reachable workspace (Monte-Carlo)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
    return fig


def arm_pose_svg(theta, width=400, height=400):
    """Render the arm pose as a standalone SVG string.

    The arm is drawn in a viewBox spanning [-1.35, 1.35] m in both axes
    (y flipped, SVG y grows downwards).  Links are blue lines, joints are
    white-filled circles, the base is a dark square and the end-effector
    orientation is a green arrow.
    """
    positions, (x, y, phi) = forward_kinematics(theta)
    reach = max_reach()
    pad = 0.15
    span = 2.0 * (reach + pad)

    def to_svg(px, py):
        sx = (px + reach + pad) / span * width
        sy = (reach + pad - py) / span * height
        return sx, sy

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f7f7"/>',
    ]
    # Reach circle.
    cx0, cy0 = to_svg(0, 0)
    r_px = reach / span * width
    parts.append(
        f'<circle cx="{cx0:.1f}" cy="{cy0:.1f}" r="{r_px:.1f}" fill="none" '
        f'stroke="#999" stroke-dasharray="5,5"/>'
    )
    # Links.
    for (x1, y1), (x2, y2) in zip(positions[:-1], positions[1:]):
        sx1, sy1 = to_svg(x1, y1)
        sx2, sy2 = to_svg(x2, y2)
        parts.append(
            f'<line x1="{sx1:.1f}" y1="{sy1:.1f}" x2="{sx2:.1f}" '
            f'y2="{sy2:.1f}" stroke="#1f77b4" stroke-width="5" '
            f'stroke-linecap="round"/>'
        )
    # End-effector orientation arrow.
    ex, ey = to_svg(x, y)
    ax_, ay_ = to_svg(x + 0.15 * math.cos(phi), y + 0.15 * math.sin(phi))
    parts.append(
        f'<line x1="{ex:.1f}" y1="{ey:.1f}" x2="{ax_:.1f}" y2="{ay_:.1f}" '
        f'stroke="green" stroke-width="3" marker-end="url(#arrow)"/>'
    )
    parts.append(
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" '
        'refX="6" refY="3" orient="auto">'
        '<path d="M0,0 L6,3 L0,6 z" fill="green"/></marker></defs>'
    )
    # Joints (base square first, then circles).
    bx, by = to_svg(0, 0)
    parts.append(
        f'<rect x="{bx - 7:.1f}" y="{by - 7:.1f}" width="14" height="14" '
        f'fill="#333"/>'
    )
    for px, py in positions[1:]:
        sx, sy = to_svg(px, py)
        parts.append(
            f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="7" fill="white" '
            f'stroke="#1f77b4" stroke-width="3"/>'
        )
    parts.append("</svg>")
    return "\n".join(parts)
