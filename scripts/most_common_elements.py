# the reason why we have this file is because we want to only train the model on the most common elements in the dataset
# so we maximize the number of trainin gsamples for as few elements (so the model is more accurate for a small number of elements)

import h5py
from tqdm import tqdm
from collections import Counter

IN_TRAIN_DIR = "datasets/real_mace/train"

def get_entries(element_cnt, in_dir, file_name):
    with h5py.File(f"{in_dir}/{file_name}.h5", 'r') as hdf5_file:
        num_configs = len(hdf5_file["config_batch_0"])
        for i in tqdm(range(num_configs)):
            config_group = hdf5_file[f'config_batch_0/config_{i}']
            atomic_numbers = config_group['atomic_numbers'][:]
            element_cnt.update(atomic_numbers)

def main():
    element_cnt = Counter()
    for i in range(64):
        get_entries(element_cnt, IN_TRAIN_DIR, f"train_{i}")
    print(element_cnt)

if __name__ == "__main__":
    main()