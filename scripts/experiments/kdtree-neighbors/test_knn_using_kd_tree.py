import numpy as np
import torch
import faiss

class KNNUsingKDTree:
    def __init__(self, k=12, radius=1.0):
        self.k = k
        self.radius = radius

    def knn_using_kd_tree(self, frac_coords, lattice_matrix):
        # Convert fractional coordinates to Cartesian coordinates
        cart_coords = np.dot(frac_coords, lattice_matrix)

        # Create and add vectors to the index
        index = faiss.IndexFlatL2(3)  # 3D space
        index = faiss.index_cpu_to_all_gpus(index)
        index.add(cart_coords.astype(np.float32))

        # Perform radius search
        lims, D, I = index.range_search(cart_coords.astype(np.float32), self.radius**2)

        # Filter out self-connections and prepare edge data
        edge_index_list = []
        edge_distance_list = []
        edge_distance_vec_list = []

        for i in range(len(cart_coords)):
            start, end = lims[i], lims[i+1]
            if start == end:
                continue
            
            neighbors = I[start:end]
            distances = D[start:end]

            # Remove self-connection
            mask = neighbors != i
            neighbors = neighbors[mask]
            distances = distances[mask]

            # Sort by distance and take top-k
            if len(neighbors) > self.k:
                sorted_indices = np.argsort(distances)[:self.k]
                neighbors = neighbors[sorted_indices]
                distances = distances[sorted_indices]

            edge_index_list.extend([(i, n) for n in neighbors])
            edge_distance_list.extend(distances)

            source_coords = np.tile(cart_coords[i], (len(neighbors), 1))
            target_coords = cart_coords[neighbors]
            edge_distance_vec_list.extend(target_coords - source_coords)

        # Create tensors
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()
        edge_distance = torch.tensor(edge_distance_list, dtype=torch.float)
        edge_distance_vec = torch.tensor(edge_distance_vec_list, dtype=torch.float)

        return edge_index, edge_distance, edge_distance_vec