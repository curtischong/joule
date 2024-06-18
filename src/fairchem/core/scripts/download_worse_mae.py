import heapq
import os
import numpy as np
import torch
import json

# TODO: yeet the heap and just store the 50 worse predictions from the first 50 batches of the epoch.
energy_heap = []
force_heap = []
heap_size_limit = 50

def process_loss_values(*, energy_losses, forces_losses, pred_energies, pred_forces, batch):
    # get the worse energy prediction in the batch
    worse_idx = energy_losses.argmax()
    energy_loss = energy_losses[worse_idx].item() # break the computational graph, so it's not retained
    pred_energy = pred_energies[worse_idx].item()

    dataset_path = batch.dataset_path[worse_idx]
    data_idx = batch.data_idx[worse_idx].item()

    worse_energy = (energy_loss, pred_energy, dataset_path, data_idx)
    heapq.heappush(energy_heap, worse_energy)

    # get the worse forces prediction in the batch
    worse_idx = forces_losses.argmax()
    forces_loss = forces_losses[worse_idx].item() # break the computational graph, so it's not retained

    # the start and end idx for the sample's forces
    start = torch.sum(batch.natoms[0: worse_idx])
    end = start + batch.natoms[worse_idx]
    pred_forces = pred_forces[start:end].detach()
    
    dataset_path = batch.dataset_path[worse_idx]
    data_idx = batch.data_idx[worse_idx].item()

    force_data = (forces_loss, pred_forces, dataset_path, data_idx)
    heapq.heappush(force_heap, force_data)

    if len(energy_heap) > heap_size_limit:
        heapq.heappop(energy_heap)
    if len(force_heap) > heap_size_limit:
        heapq.heappop(force_heap)

def heap_is_not_empty():
    return bool(energy_heap) or bool(force_heap)

def tuple_to_dict(keys, tuple):
    d = {}
    for key, value in zip(keys, tuple):
        if isinstance(value, torch.Tensor):
            value = value.tolist()
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        else:
            value = value
        d[key] = value
    return d

def download_heap(epoch):
    target_folder = os.path.join("out", "worst_mae")
    os.makedirs(target_folder, exist_ok=True)
    filename = os.path.join(target_folder, f"worse_preds_epoch_{epoch}.json")
    
    energy_keys = ["loss", "pred_energy", "dataset_path", "data_idx"]
    force_keys = ["loss", "pred_forces", "dataset_path", "data_idx"]

    energy_heap_as_dict = [tuple_to_dict(energy_keys, entry) for entry in energy_heap ]
    force_heap_as_dict = [tuple_to_dict(force_keys, entry) for entry in force_heap ]

    heap_contents = {
        "worse_energy": energy_heap_as_dict,
        "worse_forces": force_heap_as_dict
    }

    #np.save(filename, heap_contents)
    with open(filename, "w") as file:
        json.dump(heap_contents, file, indent=4)
        
def clear_heap():
    global energy_heap, force_heap
    energy_heap.clear()
    force_heap.clear()