import h5py
from tqdm import tqdm
most_common_elements = set([8, 1, 9, 16, 15, 7, 3, 12, 6, 14, 17, 26, 34, 5, 25, 13, 11, 23, 19, 27])


IN_TRAIN_DIR = "datasets/real_mace/train"

def cnt_systems_that_satisfy(in_dir, file_name):
    cnt = 0
    with h5py.File(f"{in_dir}/{file_name}.h5", 'r') as hdf5_file:
        num_configs = len(hdf5_file["config_batch_0"])
        for i in tqdm(range(num_configs)):
            config_group = hdf5_file[f'config_batch_0/config_{i}']
            atomic_numbers = config_group['atomic_numbers'][:]
            if all([element in most_common_elements for element in atomic_numbers]):
                cnt += 1
    return cnt


def main():
    num_systems = 0
    for i in range(64):
        num_systems += cnt_systems_that_satisfy(IN_TRAIN_DIR, f"train_{i}")
    print(num_systems)

if __name__ == "__main__":
    main()
