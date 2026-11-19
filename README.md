# roboarm-sim

A 3-DOF planar robot-arm simulator in pure Python (NumPy + Matplotlib):
forward/inverse kinematics, Jacobian, smooth joint-space trajectories,
Monte-Carlo workspace sampling, link–obstacle collision checks, and
Matplotlib/SVG visualisation — all behind a small CLI.

## What it does

- **Forward kinematics** for a 3-link planar RRR arm (base at the origin).
- **Analytic 2-link inverse kinematics** (law of cosines, elbow-up/down).
- **Iterative 3-joint IK** via a damped Jacobian pseudo-inverse
  (Levenberg–Marquardt style), tolerance 1e-6, up to 200 iterations.
- **Trajectory planning**: cubic joint-space profiles with zero velocity
  at both ends, plus Cartesian waypoint following (warm-started IK).
- **Workspace analysis**: Monte-Carlo reachability sampling and
  segment–circle collision detection against circular obstacles.
- **Plotting**: workspace scatter + arm pose (PNG, Agg backend) and an
  SVG rendering of any arm pose.

## Denavit–Hartenberg parameters

Planar special case of the DH convention: `alpha_i = 0`, `d_i = 0`,
`a_i = L_i`, joint variable `theta_i` (relative to the previous link).

| i | a_i (m) | alpha_i | d_i | theta_i |
|---|---------|---------|-----|---------|
| 1 | 0.5     | 0       | 0   | θ₁      |
| 2 | 0.4     | 0       | 0   | θ₂      |
| 3 | 0.3     | 0       | 0   | θ₃      |

Link lengths: `L1 = 0.5 m`, `L2 = 0.4 m`, `L3 = 0.3 m`
(max reach `L1 + L2 + L3 = 1.2 m`).

## Formulas

Forward kinematics (absolute link orientation `φᵢ = θ₁ + … + θᵢ`):

```
p₀ = (0, 0)
pᵢ₊₁ = pᵢ + Lᵢ · (cos φᵢ, sin φᵢ)
EE pose = (p₃, φ₃)
```

Jacobian (pose `(x, y, φ)` vs joint angles):

```
dx/dθⱼ = −Σᵢ≥ⱼ Lᵢ sin φᵢ
dy/dθⱼ =  Σᵢ≥ⱼ Lᵢ cos φᵢ
dφ/dθⱼ = 1
```

Analytic 2-link IK (law of cosines, wrist at distance `r` from base):

```
cos θ₂ = (r² − L1² − L2²) / (2·L1·L2)      (θ₂ > 0 elbow-up, θ₂ < 0 elbow-down)
θ₁ = atan2(y, x) − atan2(L2 sin θ₂, L1 + L2 cos θ₂)
```

Reachable iff `|L1 − L2| ≤ r ≤ L1 + L2`; otherwise `ValueError`.

Trajectory (cubic profile, normalised time `s = t/T ∈ [0, 1]`):

```
q(s) = q₀ + (q₁ − q₀)·(3s² − 2s³)     → q(0) = q₀, q(T) = q₁ exactly
q̇(s) = (q₁ − q₀)·(6s − 6s²)/T        → q̇(0) = q̇(T) = 0, peak 1.5·Δq/T
```

Collision: a link collides with a circular obstacle `(cx, cy, r)` iff the
shortest distance from the obstacle centre to any link segment is `≤ r`
(closest-point-on-segment test).

## Install

```bash
python3 -m pip install -r requirements.txt   # numpy, matplotlib
```

Requires Python ≥ 3.10. No other dependencies; tests use the `unittest`
stdlib module.

## Run

```bash
python3 main.py fk --angles 0 0 0
python3 main.py fk --angles 0.5 -0.3 0.8
python3 main.py ik --x 0.7 --y 0.3
python3 main.py ik --x 0.7 --y 0.3 --elbow-down
python3 main.py simulate --start 0 0 0 --goal 0.8 -0.5 0.6 \
    --duration 2 --steps 50 --csv trajectory.csv
python3 main.py plot --out workspace.png --svg arm.svg
```

`python3 -m roboarm` also works (same CLI via `roboarm/cli.py`).

## Module overview

| Module | Contents |
|--------|----------|
| `roboarm/kinematics.py` | `L1/L2/L3`, `forward_kinematics`, `jacobian`, `inverse_kinematics_2link`, `inverse_kinematics_numeric` |
| `roboarm/trajectory.py` | `plan_trajectory` (cubic profile), `cartesian_path` (IK along waypoints) |
| `roboarm/workspace.py` | `sample_workspace`, `check_collision`, `reachable`, `max_reach` |
| `roboarm/plotting.py` | `plot_workspace`, `plot_arm`, `arm_pose_svg` (Matplotlib Agg) |
| `roboarm/cli.py` | argparse CLI (`fk`, `ik`, `simulate`, `plot`) |
| `main.py` | CLI entry point |
| `tests/test_roboarm.py` | `unittest` suite (FK known poses, IK round-trips, trajectory, collisions, workspace, SVG) |

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## License

MIT.
