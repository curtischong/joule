import h5py
from tqdm import tqdm

# when we say "x" satisfy this, we're not including the validation dataset. Also these numbers are WITH duplicates (i.e. there may be multiple systems with the same atoms, atomic numbers, and cell)
most_common_elements = set([8, 1, 9, 16, 15, 7, 3, 12, 6, 14, 17, 26, 34, 5, 25, 13, 11, 23, 19, 27]) # 305864 satisfy this
curtis_most_important =set([5, 6, 7, 8, 11, 12, 13, 14, 15, 20, 22, 24, 26, 27, 29, 30, 35, 47, 50, 82]) # 137944 satisfy this
most_common_elements_modified = set([8, 1, 9, 16, 15, 7, 3, 12, 6, 14, 17, 26, 34, 5, 25, 13, 11, 30, 19, 29]) # 255899 satisfy this

most_common_elements_only_one_per_sample = [8, 3, 15, 12, 16, 1, 25, 7, 26, 14, 9, 6, 29, 27, 11, 23, 19, 20, 13, 17] # 324433 satisfy this

 # if you filter out samples with > 50 elements, 251553 satisfy this
 # if you filter out samples with > 70 elements, 283385 satisfy this
max_atom_cnt = 70


allowed_elements = most_common_elements_only_one_per_sample
print("num allowed: ", len(allowed_elements))



IN_TRAIN_DIR = "datasets/real_mace/train"

def cnt_systems_that_satisfy(in_dir, file_name):
    cnt = 0
    with h5py.File(f"{in_dir}/{file_name}.h5", 'r') as hdf5_file:
        num_configs = len(hdf5_file["config_batch_0"])
        for i in tqdm(range(num_configs)):
            config_group = hdf5_file[f'config_batch_0/config_{i}']
            atomic_numbers = config_group['atomic_numbers'][:]
            if all([element in allowed_elements for element in atomic_numbers]) and max_atom_cnt != -1 and len(atomic_numbers) <= max_atom_cnt:
                cnt += 1
    return cnt


def main():
    num_systems = 0
    for i in range(64):
        cnt = cnt_systems_that_satisfy(IN_TRAIN_DIR, f"train_{i}")
        num_systems += cnt
        print("i: ", i, "cnt: ", cnt, "total: ", num_systems)

if __name__ == "__main__":
    main()
