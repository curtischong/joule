import numpy as np
import torch
import faiss

class KNNUsingKDTree:
    def __init__(self, k, cutoff_radius=1.0):
        self.k = k
        self.cutoff_radius = cutoff_radius

    def knn_using_kd_tree(self, frac_coords, lattice_matrix):
        # Convert fractional coordinates to Cartesian coordinates
        cart_coords = np.dot(frac_coords, lattice_matrix)

        # Create and add vectors to the index
        index = faiss.IndexFlatL2(3)  # 3D space
        index = faiss.index_cpu_to_all_gpus(index)
        index.add(cart_coords.astype(np.float32))

        # Perform KNN search
        distances, indices = index.search(cart_coords.astype(np.float32), self.k)

        # Create edge_index
        row = np.repeat(np.arange(indices.shape[0]), indices.shape[1])
        col = indices.flatten()

        # Create edge_distance
        edge_distance = distances.flatten()

        # Filter for all edges with distance < cutoff_radius
        # since the number of nearest neighbors (k) is small, masking is fast
        mask = edge_distance < self.cutoff_radius**2  # squared distance
        row = row[mask]
        col = col[mask]
        edge_distance = edge_distance[mask]

        # Create edge_index tensor
        edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)

        # Create edge_distance tensor
        edge_distance = torch.tensor(np.sqrt(edge_distance), dtype=torch.float)  # convert back to actual distance

        # Create edge_distance_vec
        source_coords = cart_coords[row]
        target_coords = cart_coords[col]
        edge_distance_vec = torch.tensor(target_coords - source_coords, dtype=torch.float)

        return edge_index, edge_distance, edge_distance_vec

# Usage example:
# knn = KNNUsingKDTree(k=12, cutoff_radius=1.0)
# edge_index, edge_distance, edge_distance_vec = knn.knn_using_kd_tree(frac_coords, lattice_matrix)