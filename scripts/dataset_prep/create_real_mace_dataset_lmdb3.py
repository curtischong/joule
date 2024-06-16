import lmdb
import pickle
from tqdm import tqdm
import torch
import os
import time
import h5py
from torch_geometric.data import Data
from dataset_prep_common import generate_ranges
import random

IN_TRAIN_DIR = "datasets/real_mace/train"
IN_VAL_DIR = "datasets/real_mace/val"

OUT_TRAIN_DIR = "datasets/lmdb/real_mace3/train"
OUT_VAL_DIR = "datasets/lmdb/real_mace3/val"
OUT_TEST_DIR = "datasets/lmdb/real_mace3/test"

max_rows_in_output_lmdb = 50000
MAX_NUM_ATOMS = 70

most_common_elements_only_one_per_sample = [8, 3, 15, 12, 16, 1, 25, 7, 26, 14, 9, 6, 29, 27, 11, 23, 19, 20, 13, 17] 


def main():
    os.makedirs(OUT_TRAIN_DIR, exist_ok=True)
    os.makedirs(OUT_VAL_DIR, exist_ok=True)
    os.makedirs(OUT_TEST_DIR, exist_ok=True)


    results = parse_datasets(IN_VAL_DIR, "val", num_files=64)
    results.extend(parse_datasets(IN_TRAIN_DIR, "train", num_files=64))
    results = remove_duplicates(results)
    random.shuffle(results) # shuffle AFTER we remove duplicates so we can track the original index of the duplicated samples

    range = generate_ranges(len(results), split_frac=[0.7, 0.15, 0.15], start_at_1=False)

    train_range = range[0]
    val_range = range[1]
    test_range = range[2]

    create_lmdb(OUT_TRAIN_DIR, train_range, results)
    create_lmdb(OUT_VAL_DIR, val_range, results)
    create_lmdb(OUT_TEST_DIR, test_range, results)

def remove_duplicates(results):
    # filter out duplicated samples
    # if there is a duplicated sample, NONE of them will be kept.
    # This is because we don't know which is the correct one, so we assume both are incorrect

    duplicated_results = set() # keep them in a set since more than one duplicate can exist
    num_duplicates = 0
    unique_results = {}
    for i, res in enumerate(results):
        hash = str(res["positions"]) + str(res["atomic_numbers"]) + str(res["cell"])
        if hash in unique_results:
            print(f"duplicate found at {i}")
            # print(hash)
            num_duplicates += 1
            duplicated_results.add(hash)
        else:
            unique_results[hash] = res

    for duplicate in duplicated_results:
        del unique_results[duplicate]

    print(f"found {num_duplicates} duplicates")
    print("num previous results: ", len(results))
    print("num cleaned results: ", len(unique_results))
    return list(unique_results.values())


def create_lmdb(dataset_path, dataset_range, atoms: list[any]):
    range_start = dataset_range[0]
    range_end = dataset_range[1]
    ith_loop = 0
    while range_start < range_end:
        current_segment_end = min(range_start + max_rows_in_output_lmdb, range_end)
        create_single_lmdb(f"{dataset_path}/{ith_loop}", atoms[range_start:current_segment_end])
        range_start += max_rows_in_output_lmdb
        ith_loop += 1

# It's faster to read them one by one than to parellelize this
def parse_datasets(in_dir, in_dir_prefix, num_files):
    results = []
    for i in range(num_files):
        print(f"parsing {in_dir_prefix}_{i}")
        results.extend(get_entries(in_dir, f"{in_dir_prefix}_{i}"))
    return results

# this should be a list of pymatgen.io.ase.MSONAtoms
def create_single_lmdb(dataset_path, atoms: list[any]):
    db = lmdb.open(
        f"{dataset_path}.lmdb",
        map_size=1099511627776 * 2, # two terabytes is the max size of the db
        subdir=False,
        meminit=False,
        map_async=True,
    )

    start_time = time.time()

    print(f"reading {dataset_path}")
    num_samples = len(atoms)

    for fid, data in tqdm(enumerate(atoms), total=num_samples):
        positions = torch.Tensor(data["positions"])
        cell = torch.Tensor(data["cell"]).view(1, 3, 3)
        atomic_numbers = torch.Tensor(data["atomic_numbers"])
        natoms = positions.shape[0]

        data = Data(
            cell=cell,
            pos=positions,
            atomic_numbers=atomic_numbers,
            natoms=natoms,
            fid=torch.LongTensor([fid]),
            fixed=torch.full((natoms,), 0, dtype=torch.int8), # make all the atoms fixed, so the model's prediction for each atom contributes to the loss
        )

        txn = db.begin(write=True)
        txn.put(f"{fid}".encode("ascii"), pickle.dumps(data, protocol=-1))
        txn.commit()

    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(num_samples, protocol=-1))
    txn.commit()


    db.sync()
    db.close()

    end_time = time.time()
    print(f"{dataset_path} lmdb created")
    print(f"Time to create lmdb: {end_time - start_time}")


# this file is just for debugging
def print_all_groups(hdf5_file, group_name='/'):
    group = hdf5_file[group_name]
    for key in group.keys():
        if isinstance(group[key], h5py.Group):
            print(f"Group: {group_name}{key}")
            print_all_groups(hdf5_file, f"{group_name}{key}/")

def get_entries(in_dir, file_name):
    entries = []

    with h5py.File(f"{in_dir}/{file_name}.h5", 'r') as hdf5_file:
        num_configs = len(hdf5_file["config_batch_0"])
        for i in tqdm(range(num_configs)):
            config_group = hdf5_file[f'config_batch_0/config_{i}']
            atomic_numbers = config_group['atomic_numbers'][:]

            # filter out samples
            if len(atomic_numbers) > MAX_NUM_ATOMS or not all([element in most_common_elements_only_one_per_sample for element in atomic_numbers]):
                continue

            properties = {
                "atomic_numbers": atomic_numbers,
                "cell":config_group['cell'][:],
                "energy": config_group['energy'][()], # curtis: energy () since we're getting a scalar value, not a tensor
                "forces": config_group['forces'][:],
                "positions": config_group['positions'][:],
            }
            entries.append(properties)

    print(f"found {num_configs} systems")
    print(f"after filtering, found {len(entries)} systems")
    return entries

if __name__ == "__main__":
    main()