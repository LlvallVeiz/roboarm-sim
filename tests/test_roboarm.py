"""Unit tests for the roboarm 3-DOF planar arm simulator."""

import math
import unittest

import numpy as np

from roboarm.kinematics import (
    L1,
    L2,
    L3,
    forward_kinematics,
    inverse_kinematics_2link,
    inverse_kinematics_numeric,
    jacobian,
)
from roboarm.plotting import arm_pose_svg
from roboarm.trajectory import cartesian_path, plan_trajectory
from roboarm.workspace import check_collision, reachable, sample_workspace


class TestForwardKinematics(unittest.TestCase):
    def test_all_zero_angles(self):
        """All-zero angles: arm stretched along +x, EE at (L1+L2+L3, 0)."""
        positions, pose = forward_kinematics([0, 0, 0])
        self.assertEqual(positions[0], (0.0, 0.0))
        self.assertAlmostEqual(positions[1][0], L1)
        self.assertAlmostEqual(positions[1][1], 0.0)
        self.assertAlmostEqual(positions[2][0], L1 + L2)
        self.assertAlmostEqual(positions[3][0], L1 + L2 + L3)
        self.assertAlmostEqual(positions[3][1], 0.0)
        self.assertAlmostEqual(pose[0], L1 + L2 + L3)
        self.assertAlmostEqual(pose[1], 0.0)
        self.assertAlmostEqual(pose[2], 0.0)

    def test_first_joint_90_degrees(self):
        """90 deg on joint 1: arm stretched along +y."""
        positions, pose = forward_kinematics([math.pi / 2, 0, 0])
        self.assertAlmostEqual(pose[0], 0.0, places=12)
        self.assertAlmostEqual(pose[1], L1 + L2 + L3)
        self.assertAlmostEqual(pose[2], math.pi / 2)
        self.assertAlmostEqual(positions[1][1], L1)

    def test_folded_arm(self):
        """theta = (0, pi, 0): link 2 folds back over link 1."""
        _, pose = forward_kinematics([0, math.pi, 0])
        # Link 1: +x by L1; link 2: -x by L2; link 3: -x by L3.
        self.assertAlmostEqual(pose[0], L1 - L2 - L3)
        self.assertAlmostEqual(pose[1], 0.0, places=12)
        self.assertAlmostEqual(pose[2], math.pi)

    def test_pose_orientation_is_angle_sum(self):
        _, pose = forward_kinematics([0.3, -0.5, 1.1])
        self.assertAlmostEqual(pose[2], 0.3 - 0.5 + 1.1)

    def test_bad_input_shape(self):
        with self.assertRaises(ValueError):
            forward_kinematics([0, 0])


class TestJacobian(unittest.TestCase):
    def test_matches_finite_differences(self):
        theta = [0.4, -0.7, 1.0]
        J = jacobian(theta)
        self.assertEqual(J.shape, (3, 3))
        eps = 1e-7
        for j in range(3):
            dq = np.zeros(3)
            dq[j] = eps
            _, p_plus = forward_kinematics(np.array(theta) + dq)
            _, p_minus = forward_kinematics(np.array(theta) - dq)
            fd = (np.array(p_plus) - np.array(p_minus)) / (2 * eps)
            # Orientation row is exactly 1 for every joint.
            np.testing.assert_allclose(J[:, j], fd, atol=1e-5)

    def test_zero_config_jacobian(self):
        J = jacobian([0, 0, 0])
        # dx/dtheta_j = 0, dy/dtheta_j = sum of remaining link lengths.
        np.testing.assert_allclose(J[0], [0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(
            J[1], [L1 + L2 + L3, L2 + L3, L3], atol=1e-12)
        np.testing.assert_allclose(J[2], [1, 1, 1], atol=1e-12)


class TestInverseKinematics2Link(unittest.TestCase):
    TARGETS = [(0.7, 0.3), (0.2, 0.8), (0.85, -0.1), (0.5, 0.0),
               (0.1, 0.6)]

    def _check_round_trip(self, x, y, elbow_up):
        t1, t2 = inverse_kinematics_2link(x, y, elbow_up=elbow_up)
        positions, _ = forward_kinematics([t1, t2, 0.0])
        wrist = positions[2]  # end of link 2
        self.assertAlmostEqual(wrist[0], x, places=6)
        self.assertAlmostEqual(wrist[1], y, places=6)

    def test_round_trip_elbow_up(self):
        for x, y in self.TARGETS:
            with self.subTest(target=(x, y)):
                self._check_round_trip(x, y, elbow_up=True)

    def test_round_trip_elbow_down(self):
        for x, y in self.TARGETS:
            with self.subTest(target=(x, y)):
                self._check_round_trip(x, y, elbow_up=False)

    def test_branches_differ(self):
        t_up = inverse_kinematics_2link(0.7, 0.3, elbow_up=True)
        t_down = inverse_kinematics_2link(0.7, 0.3, elbow_up=False)
        self.assertNotAlmostEqual(t_up[1], t_down[1])
        self.assertGreater(t_up[1], 0)
        self.assertLess(t_down[1], 0)

    def test_max_reach_boundary(self):
        # Straight out along +x at full 2-link stretch.
        t1, t2 = inverse_kinematics_2link(L1 + L2, 0.0)
        self.assertAlmostEqual(t1, 0.0, places=6)
        self.assertAlmostEqual(t2, 0.0, places=6)

    def test_unreachable_too_far(self):
        with self.assertRaises(ValueError):
            inverse_kinematics_2link(L1 + L2 + 0.01, 0.0)

    def test_unreachable_too_close(self):
        with self.assertRaises(ValueError):
            inverse_kinematics_2link(0.0, 0.0)  # r=0 < |L1-L2| = 0.1

    def test_unreachable_diagonal(self):
        with self.assertRaises(ValueError):
            inverse_kinematics_2link(1.0, 1.0)  # r ~ 1.41 > 0.9


class TestInverseKinematicsNumeric(unittest.TestCase):
    def test_converges_to_target_pose(self):
        target = (0.8, 0.4, 0.5)
        theta = inverse_kinematics_numeric(target, [0.1, 0.1, 0.1])
        _, pose = forward_kinematics(theta)
        err = np.array(pose) - np.array(target)
        err[2] = (err[2] + math.pi) % (2 * math.pi) - math.pi
        self.assertLess(np.linalg.norm(err), 1e-6)

    def test_converges_from_different_seeds(self):
        # Target generated from a known configuration, so it is
        # guaranteed reachable (wrist distance from base < L1 + L2).
        _pos, target = forward_kinematics([0.9, -0.6, 1.2])
        for seed in ([0, 0, 0], [1.0, -1.0, 0.5], [-0.5, 0.5, -0.5]):
            with self.subTest(seed=seed):
                theta = inverse_kinematics_numeric(target, seed)
                _, pose = forward_kinematics(theta)
                self.assertAlmostEqual(pose[0], target[0], places=6)
                self.assertAlmostEqual(pose[1], target[1], places=6)
                dphi = (pose[2] - target[2] + math.pi) % (2 * math.pi) \
                    - math.pi
                self.assertAlmostEqual(dphi, 0.0, places=6)

    def test_unreachable_target_raises(self):
        # Far outside max reach (1.2 m).
        with self.assertRaises(ValueError):
            inverse_kinematics_numeric((3.0, 0.0, 0.0), [0, 0, 0],
                                       max_iter=50)


class TestTrajectory(unittest.TestCase):
    def test_endpoints_exact(self):
        q0 = np.array([0.0, 0.0, 0.0])
        q1 = np.array([0.5, -0.8, 1.2])
        traj = plan_trajectory(q0, q1, duration=2.0, steps=20)
        self.assertEqual(len(traj), 21)
        np.testing.assert_array_equal(traj[0]["q"], q0)
        np.testing.assert_array_equal(traj[-1]["q"], q1)
        self.assertAlmostEqual(traj[0]["t"], 0.0)
        self.assertAlmostEqual(traj[-1]["t"], 2.0)

    def test_zero_end_velocities(self):
        traj = plan_trajectory([0, 0, 0], [1.0, 0.5, -0.5],
                               duration=1.5, steps=30)
        np.testing.assert_array_almost_equal(traj[0]["qd"], [0, 0, 0])
        np.testing.assert_array_almost_equal(traj[-1]["qd"], [0, 0, 0])
        # Mid-trajectory velocity is non-zero (bell-shaped profile).
        mid = traj[len(traj) // 2]["qd"]
        self.assertGreater(np.linalg.norm(mid), 0.1)

    def test_peak_velocity_is_1p5_times_average(self):
        # Cubic profile peak velocity = 1.5 * dq / T at s = 0.5.
        q0 = np.array([0.0, 0.0, 0.0])
        q1 = np.array([1.0, 0.0, 0.0])
        traj = plan_trajectory(q0, q1, duration=2.0, steps=100)
        peak = max(e["qd"][0] for e in traj)
        self.assertAlmostEqual(peak, 1.5 * 1.0 / 2.0, places=2)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            plan_trajectory([0, 0, 0], [1, 1, 1], duration=0, steps=10)
        with self.assertRaises(ValueError):
            plan_trajectory([0, 0], [1, 1, 1], duration=1, steps=10)

    def test_cartesian_path_follows_waypoints(self):
        waypoints = [(0.9, 0.2, 0.3), (0.7, 0.5, 0.6), (0.4, 0.7, 1.0)]
        configs = cartesian_path(waypoints)
        self.assertEqual(len(configs), len(waypoints))
        for target, q in zip(waypoints, configs):
            _, pose = forward_kinematics(q)
            self.assertAlmostEqual(pose[0], target[0], places=5)
            self.assertAlmostEqual(pose[1], target[1], places=5)


class TestWorkspace(unittest.TestCase):
    def test_sample_count_and_bounds(self):
        n = 500
        points = sample_workspace(n, seed=123)
        self.assertEqual(points.shape, (n, 2))
        radii = np.hypot(points[:, 0], points[:, 1])
        # No point can exceed the fully stretched arm.
        self.assertLessEqual(radii.max(), L1 + L2 + L3 + 1e-9)
        # Samples should actually spread out (not all near the origin).
        self.assertGreater(radii.mean(), 0.3)

    def test_reachable(self):
        self.assertTrue(reachable(0.5, 0.0))
        self.assertTrue(reachable(0.9, 0.0))   # max 2-link stretch
        self.assertFalse(reachable(1.5, 0.0))  # beyond L1 + L2
        self.assertFalse(reachable(0.0, 0.0))  # inside the inner annulus

    def test_collision_detected(self):
        # Arm stretched along +x; obstacle sitting on link 1.
        positions, _ = forward_kinematics([0, 0, 0])
        self.assertTrue(check_collision(positions, (0.25, 0.0, 0.05)))
        # Obstacle just off the arm, small radius: no collision.
        self.assertFalse(check_collision(positions, (0.25, 0.2, 0.05)))

    def test_collision_near_joint(self):
        # Obstacle centred on joint 2 (end of link 1).
        positions, _ = forward_kinematics([0.6, -0.4, 0.2])
        j2 = positions[1]
        self.assertTrue(check_collision(positions, (j2[0], j2[1], 0.01)))
        # Same spot but far away in y: no collision.
        self.assertFalse(
            check_collision(positions, (j2[0], j2[1] + 0.5, 0.05)))

    def test_collision_touching_counts(self):
        positions, _ = forward_kinematics([0, 0, 0])
        # Circle tangent to link 1 (centre 0.05 above, radius 0.05).
        self.assertTrue(check_collision(positions, (0.25, 0.05, 0.05)))


class TestPlotting(unittest.TestCase):
    def test_arm_pose_svg(self):
        svg = arm_pose_svg([0.5, -0.3, 0.8])
        self.assertIn("<svg", svg)
        self.assertIn("</svg>", svg)
        self.assertIn("<line", svg)
        self.assertIn("<circle", svg)


if __name__ == "__main__":
    unittest.main()
