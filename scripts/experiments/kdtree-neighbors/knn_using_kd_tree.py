import numpy as np
import torch
import faiss

class KNNUsingKDTree:
    def __init__(self, k, cutoff_radius=1.0):
        self.k = k
        self.cutoff_radius = cutoff_radius

    def create_graph(self, frac_coords, lattice_matrix):
        # Convert fractional coordinates to Cartesian coordinates
        cart_coords = np.dot(frac_coords, lattice_matrix)

        # Create extended coordinates considering periodic images
        extended_coords = self._create_extended_coords(cart_coords, lattice_matrix)

        # Create and add vectors to the index
        index = faiss.IndexFlatL2(3)  # 3D space
        # index = faiss.index_cpu_to_all_gpus(index) # TODO(curtis): enable this when running on a cluster
        index.add(extended_coords.astype(np.float32))

        # Perform KNN search
        distances, indices = index.search(cart_coords.astype(np.float32), self.k)

        # Create edge_index
        row = np.repeat(np.arange(cart_coords.shape[0]), indices.shape[1])
        col = indices.flatten() % cart_coords.shape[0]  # Wrap indices to original atoms

        # Create edge_distance
        edge_distance = distances.flatten()

        # Filter for all edges with distance < cutoff_radius
        mask = edge_distance < self.cutoff_radius**2  # squared distance
        row = row[mask]
        col = col[mask]
        edge_distance = edge_distance[mask]

        # Create edge_index tensor
        edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)

        # Create edge_distance tensor
        edge_distance = torch.tensor(np.sqrt(edge_distance), dtype=torch.float)  # convert back to actual distance

        # Create edge_distance_vec considering periodic boundary conditions
        edge_distance_vec = self._compute_periodic_distance_vec(cart_coords[row], cart_coords[col], lattice_matrix)

        # # Create edge_distance_vec
        # source_coords = cart_coords[row]
        # target_coords = cart_coords[col]
        # edge_distance_vec = torch.tensor(target_coords - source_coords, dtype=torch.float)

        return edge_index, edge_distance, edge_distance_vec

    def _create_extended_coords(self, cart_coords, lattice_matrix):
        # Create a 3x3x3 supercell
        offsets = np.array([-1, 0, 1])
        extended_coords = []
        for i in offsets:
            for j in offsets:
                for k in offsets:
                    offset = np.dot(np.array([i, j, k]), lattice_matrix)
                    extended_coords.append(cart_coords + offset)
        return np.vstack(extended_coords)

    def _compute_periodic_distance_vec(self, source_coords, target_coords, lattice_matrix):
        diff = target_coords - source_coords
        # Apply minimum image convention
        for i in range(3):
            diff[:, i] = diff[:, i] - np.round(diff[:, i] / np.linalg.norm(lattice_matrix[i])) * np.linalg.norm(lattice_matrix[i])
        return torch.tensor(diff, dtype=torch.float)

# Usage example:
# knn = KNNUsingKDTree(k=12, cutoff_radius=1.0)
# edge_index, edge_distance, edge_distance_vec = knn.knn_using_kd_tree(frac_coords, lattice_matrix)