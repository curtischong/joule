import numpy as np
import torch
from scipy.spatial import cKDTree

class KNNUsingKDTree:
    def __init__(self, k=12, self_interaction=False):
        self.k = k
        self.self_interaction = self_interaction

    def __call__(self, frac_coords, lattice_matrix):
        # Convert fractional coordinates to Cartesian coordinates
        cart_coords = np.dot(frac_coords, lattice_matrix)

        # Create periodic images
        images = np.array([[0, 0, 0], 
                           [1, 0, 0], [-1, 0, 0], 
                           [0, 1, 0], [0, -1, 0],
                           [0, 0, 1], [0, 0, -1],
                           [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
                           [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
                           [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
                           [1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1],
                           [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]])

        # Create extended structure
        extended_cart_coords = np.concatenate([cart_coords + np.dot(image, lattice_matrix) for image in images])
        extended_indices = np.concatenate([np.arange(len(cart_coords)) for _ in range(len(images))])

        # Create KD-tree
        tree = cKDTree(extended_cart_coords)

        # Query KD-tree for k nearest neighbors
        distances, indices = tree.query(cart_coords, k=self.k + 1)

        # Remove self-interactions if not desired
        if not self.self_interaction:
            distances = distances[:, 1:]
            indices = indices[:, 1:]
        else:
            distances = distances[:, :self.k]
            indices = indices[:, :self.k]

        # Map indices back to original structure
        indices = extended_indices[indices]

        # Create edge_index
        row = np.repeat(np.arange(len(frac_coords)), self.k)
        col = indices.flatten()
        edge_index = np.stack([row, col])

        # Calculate edge_distance and edge_distance_vec
        edge_distance = distances.flatten()
        edge_distance_vec = extended_cart_coords[indices.flatten()] - np.repeat(cart_coords, self.k, axis=0)

        # Apply minimum image convention
        for i in range(3):
            edge_distance_vec[:, i] = (edge_distance_vec[:, i] + lattice_matrix[i, i] / 2) % lattice_matrix[i, i] - lattice_matrix[i, i] / 2

        # Convert to PyTorch tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_distance = torch.tensor(edge_distance, dtype=torch.float)
        edge_distance_vec = torch.tensor(edge_distance_vec, dtype=torch.float)

        return edge_index, edge_distance, edge_distance_vec

def knn_using_kd_tree(frac_coords, lattice_matrix):
    knn = KNNUsingKDTree()
    return knn(frac_coords, lattice_matrix)