import unittest

import torch

from opencood.models.sub_modules.lidar_unreliability_estimator import (
    LidarUnreliabilityEstimator,
)


def make_processed_lidar(coords, num_points, map_size=(6, 8),
                         batch_size=None, with_intensity=False):
    coords = torch.tensor(coords, dtype=torch.long)
    num_points = torch.tensor(num_points, dtype=torch.float32)
    if coords.numel() == 0:
        coords = coords.reshape(0, 4)
    if batch_size is None:
        batch_size = int(coords[:, 0].max().item()) + 1 if coords.numel() else 1

    processed = {
        'voxel_coords': coords,
        'voxel_num_points': num_points,
        'record_len': torch.tensor([batch_size], dtype=torch.long),
    }
    if with_intensity:
        max_points = 4
        voxel_features = torch.zeros(coords.shape[0], max_points, 4)
        for i, count in enumerate(num_points.long().clamp(max=max_points)):
            if count > 0:
                voxel_features[i, :count, 3] = 0.5 + 0.1 * i
        processed['voxel_features'] = voxel_features
    return processed


class TestLidarUnreliabilityEstimator(unittest.TestCase):
    def make_estimator(self, mode='heuristic', **kwargs):
        args = {
            'mode': mode,
            'map_size': [6, 8],
            'pool_kernel': 3,
            'point_count_percentile': 0.95,
            'point_count_weight': 0.55,
            'occupancy_sparsity_weight': 0.35,
            'distance_weight': 0.10,
            'valid_region_mode': 'all',
        }
        args.update(kwargs)
        return LidarUnreliabilityEstimator(args)

    def test_shape_and_range_for_two_batches(self):
        estimator = self.make_estimator()
        processed = make_processed_lidar(
            coords=[
                [0, 0, 1, 1],
                [0, 0, 2, 3],
                [1, 0, 4, 5],
            ],
            num_points=[5, 8, 6],
            batch_size=2,
        )

        output = estimator(processed)

        self.assertEqual(tuple(output['U_L'].shape), (2, 1, 6, 8))
        self.assertTrue(torch.all(output['U_L'] >= 0.0))
        self.assertTrue(torch.all(output['U_L'] <= 1.0))
        self.assertIn('lidar_reliability_signals', output)

    def test_empty_lidar_cells_are_highly_unreliable(self):
        estimator = self.make_estimator()
        processed = make_processed_lidar([], [], batch_size=2)

        output = estimator(processed)

        self.assertEqual(tuple(output['U_L'].shape), (2, 1, 6, 8))
        self.assertGreater(float(output['U_L'].mean()), 0.85)
        self.assertTrue(torch.isfinite(output['U_L']).all())

    def test_dense_lidar_cells_are_more_reliable_than_empty_cells(self):
        estimator = self.make_estimator()
        dense_coords = []
        dense_counts = []
        for y in range(6):
            for x in range(8):
                dense_coords.append([0, 0, y, x])
                dense_counts.append(32)
        dense = estimator(make_processed_lidar(dense_coords, dense_counts))
        empty = estimator(make_processed_lidar([], [], batch_size=1))

        self.assertLess(float(dense['U_L'][0, 0, 3, 4]), 0.20)
        self.assertLess(float(dense['U_L'].mean()), float(empty['U_L'].mean()))

    def test_learned_variant_has_parameters_and_valid_output(self):
        estimator = self.make_estimator(mode='learned', hidden_channels=8,
                                        use_intensity=True, intensity_weight=0.1)
        processed = make_processed_lidar(
            coords=[
                [0, 0, 1, 1],
                [0, 0, 2, 3],
                [0, 0, 4, 6],
            ],
            num_points=[4, 6, 2],
            with_intensity=True,
        )

        output = estimator(processed)
        param_count = sum(p.numel() for p in estimator.parameters())

        self.assertGreater(param_count, 0)
        self.assertEqual(tuple(output['U_L'].shape), (1, 1, 6, 8))
        self.assertTrue(torch.all(output['U_L'] >= 0.0))
        self.assertTrue(torch.all(output['U_L'] <= 1.0))
        self.assertIn('lidar_heuristic_unreliability', output)
        self.assertTrue(output['lidar_intensity_available'])


if __name__ == '__main__':
    unittest.main()
