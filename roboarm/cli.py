"""Command-line interface for the 3-DOF planar robot-arm simulator.

Subcommands:
    fk        forward kinematics for a joint configuration
    ik        analytic 2-link inverse kinematics for a target (x, y)
    simulate  joint-space trajectory + CSV export
    plot      workspace scatter plot with a sample arm pose
"""

import argparse
import csv
import math
import sys

import numpy as np

from . import kinematics as kin
from .plotting import arm_pose_svg, plot_arm, plot_workspace
from .trajectory import cartesian_path, plan_trajectory
from .workspace import check_collision, sample_workspace


def _fmt_pose(pose):
    x, y, phi = pose
    return (f"({x:.4f}, {y:.4f}) m, phi = {phi:.4f} rad "
            f"({math.degrees(phi):.2f} deg)")


def cmd_fk(args):
    theta = [args.angles[0], args.angles[1], args.angles[2]]
    positions, pose = kin.forward_kinematics(theta)
    print("Joint angles (rad): " + ", ".join(f"{a:.4f}" for a in theta))
    print("Joint angles (deg): " + ", ".join(
        f"{math.degrees(a):.2f}" for a in theta))
    for i, (x, y) in enumerate(positions):
        label = "base " if i == 0 else f"joint {i + 1}"
        print(f"  {label:<8} ({x:.4f}, {y:.4f}) m")
    print("End-effector pose: " + _fmt_pose(pose))
    J = kin.jacobian(theta)
    print("Jacobian:")
    for row in J:
        print("  " + "  ".join(f"{v: .4f}" for v in row))
    return 0


def cmd_ik(args):
    try:
        t1, t2 = kin.inverse_kinematics_2link(
            args.x, args.y, elbow_up=not args.elbow_down)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    branch = "elbow-down" if args.elbow_down else "elbow-up"
    print(f"Target wrist position: ({args.x:.4f}, {args.y:.4f}) m [{branch}]")
    print(f"  theta_1 = {t1:.6f} rad ({math.degrees(t1):.2f} deg)")
    print(f"  theta_2 = {t2:.6f} rad ({math.degrees(t2):.2f} deg)")
    # Verify by forward kinematics of the first two links: the wrist is
    # positions[2] (end of link 2), not the end-effector pose.
    positions, _pose = kin.forward_kinematics([t1, t2, 0.0])
    wrist = positions[2]
    print(f"  check: FK wrist = ({wrist[0]:.6f}, {wrist[1]:.6f}) m")
    return 0


def cmd_simulate(args):
    q_start = np.array(args.start, dtype=float)
    q_goal = np.array(args.goal, dtype=float)
    traj = plan_trajectory(q_start, q_goal, args.duration, args.steps)
    print(f"Trajectory: {len(traj)} samples over {args.duration:.2f} s")
    print("  start: " + ", ".join(f"{a:.4f}" for a in q_start) + " rad")
    print("  goal:  " + ", ".join(f"{a:.4f}" for a in q_goal) + " rad")
    peak_v = max(float(np.max(np.abs(e["qd"]))) for e in traj)
    print(f"  peak joint velocity: {peak_v:.4f} rad/s")
    print(f"  start velocity: {np.linalg.norm(traj[0]['qd']):.2e} rad/s")
    print(f"  end   velocity: {np.linalg.norm(traj[-1]['qd']):.2e} rad/s")
    _, pose0 = kin.forward_kinematics(traj[0]["q"])
    _, pose1 = kin.forward_kinematics(traj[-1]["q"])
    print("  EE start: " + _fmt_pose(pose0))
    print("  EE goal:  " + _fmt_pose(pose1))
    with open(args.csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t", "theta1", "theta2", "theta3",
                         "dtheta1", "dtheta2", "dtheta3", "ee_x", "ee_y",
                         "ee_phi"])
        for entry in traj:
            _, pose = kin.forward_kinematics(entry["q"])
            writer.writerow([f"{entry['t']:.4f}"]
                            + [f"{v:.6f}" for v in entry["q"]]
                            + [f"{v:.6f}" for v in entry["qd"]]
                            + [f"{v:.6f}" for v in pose])
    print(f"  CSV written to {args.csv} ({len(traj)} rows)")
    return 0


def cmd_plot(args):
    points = sample_workspace(args.samples, seed=42)
    # A sample Cartesian path: a small arc of waypoints the arm can follow.
    waypoints = [(0.9, 0.2, 0.3), (0.7, 0.5, 0.6), (0.4, 0.7, 1.0),
                 (0.1, 0.8, 1.3)]
    configs = cartesian_path(waypoints)
    poses = [kin.forward_kinematics(q)[1] for q in configs]
    fig = plot_workspace(points, path=poses)
    # Overlay one arm pose on the same axes.
    plot_arm(configs[1], ax=fig.axes[0], show=False)
    fig.savefig(args.out, dpi=150)
    print(f"Workspace plot saved to {args.out} "
          f"({len(points)} samples, {len(poses)}-point path)")
    svg = arm_pose_svg(configs[1])
    if args.svg:
        with open(args.svg, "w") as fh:
            fh.write(svg)
        print(f"Arm pose SVG saved to {args.svg}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="roboarm",
        description="3-DOF planar robot-arm simulator "
                    "(forward/inverse kinematics, trajectories, workspace).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fk = sub.add_parser("fk", help="forward kinematics")
    p_fk.add_argument("--angles", nargs=3, type=float, required=True,
                      metavar=("T1", "T2", "T3"),
                      help="joint angles in radians")
    p_fk.set_defaults(func=cmd_fk)

    p_ik = sub.add_parser("ik", help="analytic 2-link inverse kinematics")
    p_ik.add_argument("--x", type=float, required=True)
    p_ik.add_argument("--y", type=float, required=True)
    p_ik.add_argument("--elbow-down", action="store_true",
                      help="pick the elbow-down branch (default elbow-up)")
    p_ik.set_defaults(func=cmd_ik)

    p_sim = sub.add_parser("simulate", help="plan a joint trajectory")
    p_sim.add_argument("--start", nargs=3, type=float, required=True,
                       metavar=("T1", "T2", "T3"))
    p_sim.add_argument("--goal", nargs=3, type=float, required=True,
                       metavar=("T1", "T2", "T3"))
    p_sim.add_argument("--duration", type=float, default=2.0)
    p_sim.add_argument("--steps", type=int, default=50)
    p_sim.add_argument("--csv", default="trajectory.csv",
                       help="output CSV path")
    p_sim.set_defaults(func=cmd_simulate)

    p_plot = sub.add_parser("plot", help="plot the reachable workspace")
    p_plot.add_argument("--out", default="workspace.png")
    p_plot.add_argument("--samples", type=int, default=2000)
    p_plot.add_argument("--svg", default=None,
                        help="also save an arm-pose SVG to this path")
    p_plot.set_defaults(func=cmd_plot)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
