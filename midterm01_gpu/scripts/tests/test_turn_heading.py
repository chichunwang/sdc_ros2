from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.integrate import cumulative_trapezoid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sensor_fusion import FusionConfig, heading_observations
from turn_heading import TurnHeadingConfig, turn_heading_observations


def synthetic(duration=20., rate_degps=3., acceleration=.4, bias_deg=3., irregular=False):
    gt = np.arange(0., duration, .25)
    if irregular:
        gt += .045*np.sin(np.arange(len(gt))*1.7)
    it = np.arange(-.1, duration+.101, .002)
    yaw = np.deg2rad(178.+rate_degps*it)
    speed = 8.+acceleration*it
    direction = np.column_stack((np.cos(yaw), np.sin(yaw)))
    path = cumulative_trapezoid(direction*speed[:, None], it, axis=0, initial=0.)
    xy = np.column_stack([np.interp(gt, it, path[:, i]) for i in range(2)])
    return gt, xy, it, yaw-np.deg2rad(bias_deg)


class TurnHeadingTests(unittest.TestCase):
    def test_exact_curved_motion_recovers_bias_with_acceleration_and_wrap(self):
        args = synthetic(irregular=True)
        rows = turn_heading_observations(*args, FusionConfig())
        accepted = [r for r in rows if r['accepted']]
        self.assertEqual(len(accepted), len(rows))
        np.testing.assert_allclose([r['offset_deg'] for r in accepted], 3., atol=2e-4)
        self.assertTrue(all(r['marginal_std_deg'] > 0 and r['lever_arm_std_deg'] > 0 for r in accepted))
        # The previous straight-course model correctly declines these turns.
        old = heading_observations(*args, FusionConfig())
        self.assertFalse(any(r['accepted'] for r in old))

    def test_time_alignment_removes_acceleration_weighting_bias(self):
        args = synthetic(duration=6., rate_degps=1.5, acceleration=2., irregular=True)
        new = turn_heading_observations(*args, FusionConfig())
        old = heading_observations(*args, FusionConfig())
        self.assertTrue(all(r['accepted'] for r in old+new))
        new_error = np.abs([r['offset_deg']-3. for r in new])
        old_error = np.abs([r['offset_deg']-3. for r in old])
        self.assertLess(np.max(new_error), .001)
        self.assertGreater(np.max(old_error), .03)

    def test_straight_course_and_large_position_origin(self):
        gt, xy, it, yaw = synthetic(rate_degps=0., acceleration=0.)
        xy += [1e6, -3e6]
        new = turn_heading_observations(gt, xy, it, yaw, FusionConfig())
        old = heading_observations(gt, xy, it, yaw, FusionConfig())
        self.assertTrue(all(r['accepted'] for r in old+new))
        np.testing.assert_allclose([r['offset_deg'] for r in new], [r['offset_deg'] for r in old], atol=1e-7)

    def test_translation_rotation_and_complete_prefix_are_invariant(self):
        gt, xy, it, yaw = synthetic(irregular=True)
        rows = turn_heading_observations(gt, xy, it, yaw, FusionConfig())
        angle = .8
        rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        shifted = turn_heading_observations(gt, xy@rotation+[1000, -3000], it, yaw+angle, FusionConfig())
        np.testing.assert_allclose([r['offset_deg'] for r in rows], [r['offset_deg'] for r in shifted], atol=1e-8)
        prefix = turn_heading_observations(gt[gt < 10], xy[gt < 10], it, yaw, FusionConfig())
        self.assertEqual(prefix, rows[:len(prefix)])

    def test_turning_lever_uncertainty_is_explicit_and_inflates(self):
        args = synthetic(rate_degps=4.)
        rows = turn_heading_observations(*args, FusionConfig())
        less = turn_heading_observations(*args, FusionConfig(), TurnHeadingConfig(lever_arm_std_m=.01))
        self.assertTrue(all(a['std_deg'] > b['std_deg'] for a, b in zip(rows, less)))
        np.testing.assert_allclose([r['offset_deg'] for r in rows], [r['offset_deg'] for r in less], atol=1e-10)
        for row in rows:
            self.assertAlmostEqual(row['lever_arm_std_deg'],
                2*row['turn_rate_rms_degps']/row['min_speed_mps'], places=10)

    def test_stop_reverse_and_endpoint_low_speed_are_rejected(self):
        gt, xy, it, yaw = synthetic(rate_degps=0., acceleration=0.)
        stopped = turn_heading_observations(gt, np.zeros_like(xy), it, yaw, FusionConfig())
        self.assertTrue(all(r['reason'] == 'low_speed' for r in stopped))
        reverse = turn_heading_observations(gt, -xy, it, yaw, FusionConfig())
        self.assertFalse(any(r['accepted'] for r in reverse))
        # Mean block speed can pass 5m/s while one endpoint is nearly stopped.
        args = synthetic(duration=2., rate_degps=0., acceleration=4.)
        args = args[0], args[1]-args[0][:, None]*np.array([np.cos(np.deg2rad(178)), np.sin(np.deg2rad(178))])*4., *args[2:]
        row = turn_heading_observations(*args, FusionConfig())[0]
        self.assertEqual(row['reason'], 'low_speed_or_reverse_within_block')

    def test_excessive_turn_and_nonrigid_motion_are_rejected(self):
        rows = turn_heading_observations(*synthetic(rate_degps=15.), FusionConfig())
        self.assertTrue(all(r['reason'] == 'turn_exceeds_local_model' for r in rows))
        gt, xy, it, yaw = synthetic(rate_degps=0.)
        xy[:, 1] += 5.*np.where(np.arange(len(gt)) % 2, 1, -1)
        rows = turn_heading_observations(gt, xy, it, yaw, FusionConfig())
        self.assertFalse(any(r['accepted'] for r in rows))

    def test_block_factor_is_local_and_does_not_leak_heldout_values(self):
        gt, xy, it, yaw = synthetic()
        rows = turn_heading_observations(gt, xy, it, yaw, FusionConfig())
        altered = xy.copy()
        affected = (gt >= 6) & (gt < 8)
        altered[affected] += [12., -8.]
        changed = turn_heading_observations(gt, altered, it, yaw, FusionConfig())
        self.assertEqual(rows[:3]+rows[4:], changed[:3]+changed[4:])

    def test_invalid_input_and_config_fail_explicitly(self):
        args = synthetic()
        for bad_gt in (args[0][::-1], np.full_like(args[0], np.nan)):
            with self.assertRaises(ValueError):
                turn_heading_observations(bad_gt, *args[1:], FusionConfig())
        with self.assertRaises(ValueError):
            turn_heading_observations(args[0], args[1][:-1], *args[2:], FusionConfig())
        with self.assertRaises(ValueError):
            turn_heading_observations(*args, FusionConfig(), TurnHeadingConfig(lever_arm_std_m=np.nan))
        with self.assertRaises(ValueError):
            turn_heading_observations(*args, replace(FusionConfig(), course_window_s=.5))


if __name__ == '__main__':
    unittest.main()
