import heapq
import os
import json

heap_size_limit = 3

energy_heap = []
force_heap = []

def process_loss_values(batch_energies, mean_batch_forces, batch_forces, batch_path, data_idx, batch_natoms):
    batch_size = batch_energies.shape[0]
    start = 0

    for i in range(batch_size):
        energy = batch_energies[i].item()
        force = mean_batch_forces[i].item()

        natoms_length = batch_natoms[i].item()
        forces_split = batch_forces[start:start + natoms_length]
        start += natoms_length
        forces_per_atom = forces_split.tolist()
        
        formatted_batch_path = str(batch_path[i])
        formatted_data_idx = data_idx[i].item()

        heapq.heappush(energy_heap, (energy, formatted_batch_path, formatted_data_idx))
        heapq.heappush(force_heap, (force, forces_per_atom, formatted_batch_path, formatted_data_idx))

        if len(energy_heap) > heap_size_limit:
            heapq.heappop(energy_heap)
        if len(force_heap) > heap_size_limit:
            heapq.heappop(force_heap)

def heap_is_not_empty():
    return bool(energy_heap) or bool(force_heap)

def download_heap(epoch):
    target_folder = os.path.join("datasets", "worst_mae")
    os.makedirs(target_folder, exist_ok=True)
    filename = os.path.join(target_folder, f"heap_contents_epoch_{epoch}.json")

    heap_contents = {
        "Energy Heap": energy_heap,
        "Force Heap": force_heap
    }

    with open(filename, "w") as file:
        json.dump(heap_contents, file)
        
def clear_heap():
    global energy_heap, force_heap
    energy_heap.clear()
    force_heap.clear()