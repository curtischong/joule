import numpy as np
import torch
import unittest
from knn_using_kd_tree import KNNUsingKDTree

class TestKNNUsingKDTree(unittest.TestCase):
    def setUp(self):
        self.knn = KNNUsingKDTree(k=4, self_interaction=False)

    def test_simple_cubic(self):
        frac_coords = np.array([
            [0, 0, 0],
            [0.5, 0.5, 0.5]
        ])
        lattice_matrix = np.eye(3) * 4.0

        edge_index, edge_distance, edge_distance_vec = self.knn(frac_coords, lattice_matrix)

        self.assertEqual(edge_index.shape, (2, 8))
        self.assertEqual(edge_distance.shape, (8,))
        self.assertEqual(edge_distance_vec.shape, (8, 3))

        expected_distance = 2.0 * np.sqrt(3)
        print("Expected distance:", expected_distance)
        print("Actual distances:", edge_distance.numpy())
        self.assertTrue(np.allclose(edge_distance.numpy(), expected_distance, atol=1e-6))

        print("Actual vectors:")
        print(edge_distance_vec.numpy())
        # Check that all vectors have the correct magnitude
        self.assertTrue(np.allclose(np.linalg.norm(edge_distance_vec.numpy(), axis=1), expected_distance, atol=1e-6))

    def test_2d_square_lattice(self):
        frac_coords = np.array([
            [0, 0, 0],
            [0.5, 0.5, 0]
        ])
        lattice_matrix = np.array([
            [3.0, 0, 0],
            [0, 3.0, 0],
            [0, 0, 1.0]
        ])

        edge_index, edge_distance, edge_distance_vec = self.knn(frac_coords, lattice_matrix)

        self.assertEqual(edge_index.shape, (2, 8))
        self.assertEqual(edge_distance.shape, (8,))
        self.assertEqual(edge_distance_vec.shape, (8, 3))

        expected_distances = np.array([1.0, 1.0, np.sqrt(2), np.sqrt(2)] * 2) * 1.5
        print("Expected distances:", expected_distances)
        print("Actual distances:", edge_distance.numpy())
        self.assertTrue(np.allclose(edge_distance.numpy(), expected_distances, atol=1e-6))

        print("Actual vectors:")
        print(edge_distance_vec.numpy())
        self.assertTrue(np.allclose(edge_distance_vec.numpy()[:, 2], 0, atol=1e-6))

    def test_self_interaction(self):
        knn_self = KNNUsingKDTree(k=5, self_interaction=True)
        frac_coords = np.array([
            [0, 0, 0],
            [0.5, 0.5, 0.5]
        ])
        lattice_matrix = np.eye(3) * 4.0

        edge_index, edge_distance, edge_distance_vec = knn_self(frac_coords, lattice_matrix)

        self.assertEqual(edge_index.shape, (2, 10))
        self.assertEqual(edge_distance.shape, (10,))
        self.assertEqual(edge_distance_vec.shape, (10, 3))

        print("Distances with self-interaction:", edge_distance.numpy())
        print("Vectors with self-interaction:")
        print(edge_distance_vec.numpy())
        self.assertTrue(0 in edge_distance.numpy())
        self.assertTrue(np.any(np.all(edge_distance_vec.numpy() == 0, axis=1)))

if __name__ == '__main__':
    unittest.main()