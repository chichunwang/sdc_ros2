from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_front_stereo import (associate_vertices, fit_image_mapping, shared_scale,
                                spatial_holdout, triangulate)
from source_rigidity import RigidityConfig


class FrontStereoTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(391)
        self.k = np.array([[420., 0, 448.], [0, 420., 294.], [0, 0, 1.]])
        self.pose = np.eye(4); self.pose[0, 3] = .8
        self.points = self.rng.uniform([-3, -2, 4], [3, 2, 15], (60, 3))

    def project(self, xyz):
        q = xyz @ self.k.T
        return q[:, :2]/q[:, 2:3]

    def test_known_metric_baseline_recovers_depth(self):
        a = self.project(self.points)
        b = self.project(self.points-self.pose[:3, 3])
        result = triangulate(a, b, self.k, self.k, self.pose)
        self.assertTrue(result['eligible'].all())
        np.testing.assert_allclose(result['xyz'], self.points, atol=1e-10)
        self.assertLess(result['reprojection_px'].max(), 1e-10)

    def test_wrong_correspondence_and_low_parallax_rejected(self):
        a = self.project(self.points)
        b = self.project(self.points-self.pose[:3, 3]); b[:10, 1] += 20
        result = triangulate(a, b, self.k, self.k, self.pose)
        self.assertFalse(result['eligible'][:10].any())
        far = self.points*100
        result = triangulate(self.project(far), self.project(far-self.pose[:3, 3]), self.k, self.k, self.pose)
        self.assertFalse(result['eligible'].any())

    def test_source_association_is_visible_vertex_and_deduplicated(self):
        c = RigidityConfig()
        uv = np.array([[224., 147.], [225, 147], [224, 148], [225, 148]])
        xyz = np.c_[np.full(4, 5.), -(uv[:, 0]-224)/180*5, -(uv[:, 1]-147)/180*5]
        # Distant point on an occluded identical ray may never be returned.
        xyz = np.r_[xyz, xyz[:1]*2]
        ids, dd = associate_vertices(xyz, np.array([[224.1, 147.1], [224.1, 147.1]]), np.eye(3), c)
        self.assertEqual(ids[0], 0); self.assertEqual(ids[1], -1)
        self.assertLess(dd[0], .2)

    def test_depth_discontinuity_prevents_association(self):
        c = RigidityConfig()
        uv = np.array([[224., 147.], [225, 147], [224, 148]])
        depth = np.array([5., 5., 20.])
        xyz = np.c_[depth, -(uv[:, 0]-224)/180*depth, -(uv[:, 1]-147)/180*depth]
        ids, _ = associate_vertices(xyz, np.array([[224., 147.]]), np.eye(3), c)
        self.assertEqual(ids[0], -1)

    def test_mapping_training_has_no_holdout_leak(self):
        c = RigidityConfig()
        pixels = self.rng.uniform([2, 2], [446, 292], (150, 2))
        h = np.array([[2., .01, 4], [0, 2., -2], [1e-5, 0, 1]])
        destination = cv2.perspectiveTransform(pixels[:, None], h)[:, 0]
        source = dict(pixels=pixels); front = dict(pixels=destination.copy())
        pairs = np.c_[np.arange(len(pixels)), np.arange(len(pixels))]
        with patch('audit_front_stereo.match_features', return_value=pairs):
            _, first, _ = fit_image_mapping(source, front, c)
            front['pixels'][spatial_holdout(pixels, (448, 294))] += [80, 0]
            _, second, _ = fit_image_mapping(source, front, c)
        self.assertTrue(first['usable']); self.assertFalse(second['usable'])
        np.testing.assert_array_equal(first['homography_front_from_virtual'], second['homography_front_from_virtual'])

    def test_shared_scale_equal_scene_weight_and_holdout_not_refit(self):
        groups = [dict(track=t, role=role, static=dict(anchors=anchors, median_log_metric_per_raw=np.log(scale)))
                  for t, role, anchors, scale in ((1, 'fit', 600, 6.), (1, 'fit', 6, 6.),
                                                   (2, 'fit', 6, 6.), (2, 'holdout', 1000, 20.))]
        first = shared_scale(groups, 'static')
        self.assertTrue(first['available']); self.assertAlmostEqual(first['ply_scale'], 6.)
        groups[-1]['static']['median_log_metric_per_raw'] = np.log(2.)
        second = shared_scale(groups, 'static')
        self.assertEqual(first['ply_scale'], second['ply_scale'])
        self.assertNotEqual(first['holdout_group_relative_scale_error'], second['holdout_group_relative_scale_error'])

    def test_one_track_never_labels_shared_calibration(self):
        groups = [dict(track=1, role='fit', static=dict(anchors=20, median_log_metric_per_raw=np.log(6.))) for _ in range(4)]
        self.assertFalse(shared_scale(groups, 'static')['available'])


if __name__ == '__main__':
    unittest.main()
