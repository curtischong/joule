import heapq
import os

heap_size_limit = 3

energy_heap = []
force_heap = []

def process_loss_values(loss_values, batch_path, data_idx):
    batch_energies, batch_forces = loss_values

    for i, (energy, force_subarray) in enumerate(zip(batch_energies, batch_forces)):
        force = sum(force_subarray) / len(force_subarray)
        formatted_batch_path = str(batch_path[i])
        formatted_data_idx = data_idx[i].item()

        heapq.heappush(energy_heap, (energy, formatted_batch_path, formatted_data_idx))
        heapq.heappush(force_heap, (force, formatted_batch_path, formatted_data_idx))

        if len(energy_heap) > heap_size_limit:
            heapq.heappop(energy_heap)
        if len(force_heap) > heap_size_limit:
            heapq.heappop(force_heap)
            print('gogogahgah')

def heap_is_not_empty():
    return bool(energy_heap) or bool(force_heap)

def download_heap(epoch):
    target_folder = os.path.join("packages", "fairchem-demo-ocpapi", "src", "fairchem", "core", "datasets", "worst_mae")
    os.makedirs(target_folder, exist_ok=True)
    filename = os.path.join(target_folder, f"heap_contents_epoch_{epoch}.txt")

    print("Target folder:", target_folder)
    print("Filename:", filename)
    print("Downloading heap contents")
    print("Energy Heap:", energy_heap)
    print("Force Heap:", force_heap)

    # target_folder = os.path.join("packages", "fairchem-demo-ocpapi", "src", "fairchem", "core", "datasets", "worst_mae")
    # os.makedirs(target_folder, exist_ok=True)
    # filename = os.path.join(target_folder, f"heap_contents_epoch_{epoch}.txt")

    # with open(filename, "w") as file:
    #     file.write("Downloading heap contents\n")
    #     file.write("Energy Heap: " + str(energy_heap) + "\n")
    #     file.write("Force Heap: " + str(force_heap) + "\n")

def clear_heap():
    global energy_heap, force_heap
    energy_heap.clear()
    force_heap.clear()