import heapq

heap_size_limit = 3

energy_heap = []
force_heap = []

def process_loss_values(loss_values):
    if len(loss_values) == 1: #When `def _compute_loss` is called, it first passes `[energy_loss]`, then later `[energy_loss, force_loss]`. We only want to extract the array when its length is 2.
        return

    energy, force = loss_values[0], loss_values[1]
    heapq.heappush(energy_heap, energy)
    heapq.heappush(force_heap, force)

    if len(energy_heap) > heap_size_limit:
        heapq.heappop(energy_heap)
    if len(force_heap) > heap_size_limit:
        heapq.heappop(force_heap)

def heap_is_not_empty():
    return bool(energy_heap) or bool(force_heap)

def download_heap():
    print("Downloading heap contents")
    print("Energy Heap:", energy_heap)
    print("Force Heap:", force_heap)

def clear_heap():
    global energy_heap, force_heap
    energy_heap.clear()
    force_heap.clear()