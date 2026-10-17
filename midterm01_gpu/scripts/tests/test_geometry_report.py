import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from accuracy.geometry_report import summarize_guided_updates, validate_geometry


class GeometryReportTests(unittest.TestCase):
    def test_candidate_rejection_is_distinct_from_returned_pose_geometry(self):
        # The returned prior may have six targets, while the rejected candidate
        # only had five. Report the recorded candidate reason without inferring
        # it from the returned pose's geometry.
        row = {'update_status': 'insufficient_unique_targets', 'map_accepted': '0',
               'candidate_inliers': '100', 'candidate_reliable_points': '20', 'candidate_unique_targets': '5',
               'reliable_points': '24', 'unique_targets': '6', 'dx': '0', 'dy': '0', 'dz': '0', 'dyaw': '0'}
        summary, errors = summarize_guided_updates([row])
        self.assertEqual(errors, [])
        self.assertEqual(summary['geometry_support_fallback_frames'], 1)
        self.assertEqual(summary['other_map_rejection_frames'], 0)
        for key, value in [('map_accepted', '1'), ('candidate_unique_targets', '6'), ('dx', '.01'),
                           ('candidate_reliable_points', '5'), ('update_status', 'not_evaluated')]:
            corrupt = dict(row)
            corrupt[key] = value
            self.assertTrue(summarize_guided_updates([corrupt])[1], key)

    def test_historical_guidance_has_no_invented_rejection_reason(self):
        summary, errors = summarize_guided_updates([{'map_accepted': '0'}])
        self.assertEqual(errors, [])
        self.assertEqual(summary, {'update_status_available': False})

    def fixture(self, directory):
        run = Path(directory)
        config = {'mode': 'diagnostic', 'normal_gain': 2., 'tangent_weight': .05,
                  'rank_absolute_threshold': .0001, 'rank_relative_threshold': .05}
        (run / 'geometry_config.json').write_text(json.dumps(config))
        stamp = 1727234445000000001
        row = {'idx': 0, 'timestamp_ns': stamp, 'map_accepted': 1, 'directional_update_used': 0,
               'input_points': 100, 'euclidean_inliers': 90, 'reliable_points': 80, 'unique_targets': 40,
               'mean_confidence': .8, 'observable_rank': 1, 'plane_rmse_m': .1,
               'span_x_m': 5, 'span_y_m': 5, 'span_z_m': 0}
        vectors = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
        for i, value in enumerate([0., 0., 0., 1.]):
            row[f'scaled_eigenvalue_{i}'] = value
        for r in range(4):
            for c in range(4):
                row[f'scaled_eigenvector_r{r}c{c}'] = vectors[r][c]
                row[f'scaled_information_r{r}c{c}'] = int(r == 2 and c == 2)
        return run, stamp, row

    def write_row(self, run, row):
        with (run / 'geometry.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    def test_valid_ground_information_keeps_integer_time_and_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            run, stamp, row = self.fixture(directory)
            self.write_row(run, row)
            summary, errors = validate_geometry(run, [stamp], 'diagnostic', 2., .05)
            self.assertEqual(errors, [])
            self.assertEqual(summary['rank_histogram']['1'], 1)
            self.assertEqual(summary['directional_update_frames'], 0)

    def test_corrupt_support_or_eigensystem_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run, stamp, row = self.fixture(directory)
            for key, value in [('reliable_points', 200), ('scaled_eigenvector_r0c0', .5),
                               ('scaled_information_r2c2', 2.), ('plane_rmse_m', 'nan'),
                               ('timestamp_ns', stamp + 1), ('directional_update_used', 1), ('observable_rank', 4)]:
                corrupt = dict(row)
                corrupt[key] = value
                self.write_row(run, corrupt)
                self.assertTrue(validate_geometry(run, [stamp], 'diagnostic', 2., .05)[1], key)


if __name__ == '__main__':
    unittest.main()
