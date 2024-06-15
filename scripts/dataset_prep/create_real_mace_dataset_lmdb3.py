from ase import Atoms
import lmdb
import pickle
from tqdm import tqdm
import torch
import os
import numpy as np
import time
from ase.calculators.singlepoint import SinglePointCalculator
import h5py
import concurrent.futures
from torch_geometric.data import Data
from scripts.dataset_prep.dataset_prep_common import get_range
import random

IN_TRAIN_DIR = "datasets/real_mace/train"
IN_VAL_DIR = "datasets/real_mace/val"

OUT_TRAIN_DIR = "datasets/lmdb/real_mace3/train"
OUT_VAL_DIR = "datasets/lmdb/real_mace3/val"
OUT_TEST_DIR = "datasets/lmdb/real_mace3/test"

max_rows_in_output_lmdb = 25000

most_common_elements_only_one_per_sample = [8, 3, 15, 12, 16, 1, 25, 7, 26, 14, 9, 6, 29, 27, 11, 23, 19, 20, 13, 17] 
MAX_JOBS = 8


def main():
    os.makedirs(OUT_TRAIN_DIR, exist_ok=True)
    os.makedirs(OUT_VAL_DIR, exist_ok=True)


    results = parse_datasets(IN_VAL_DIR, "val", num_files=64)
    results.extend(parse_datasets(IN_TRAIN_DIR, "train", num_files=64))
    random.shuffle(results)

    range = get_range(len(results), dataset_type="all")

    train_range = range[0]
    val_range = range[1]
    test_range = range[2]

    create_lmdb(OUT_TRAIN_DIR, train_range, results)
    create_lmdb(OUT_VAL_DIR, val_range, results)
    create_lmdb(OUT_TEST_DIR, test_range, results)

def create_lmdb(dataset_path, range, atoms: list[any]):
    range_start = range[0]
    range_end = range[1]
    for i in range(range_start, range_end):
        create_single_lmdb(f"{dataset_path}/{i}", atoms[i:min(i + max_rows_in_output_lmdb, range_end)])

# It's faster to read them one by one than parellelize this
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
        positions = torch.Tensor(data.get_positions())
        cell = torch.Tensor(np.array(data.get_cell())).view(1, 3, 3)
        atomic_numbers = torch.Tensor(data.get_atomic_numbers())
        natoms = positions.shape[0]

        data = Data(
            cell=cell,
            pos=positions,
            atomic_numbers=atomic_numbers,
            natoms=natoms,
            fid=torch.LongTensor([fid]),
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
        # num_configs = 1000
        for i in tqdm(range(num_configs)):
            config_group = hdf5_file[f'config_batch_0/config_{i}']
            atomic_numbers = config_group['atomic_numbers'][:]

            # filter out samples
            if not all([element in most_common_elements_only_one_per_sample for element in atomic_numbers]):
                continue


            cell = config_group['cell'][:]
            # properties["charges"].append(config_group['charges'][:])
            energy = config_group['energy'][()] # curtis: why is energy ()??
            forces = config_group['forces'][:]
            positions = config_group['positions'][:]

            # I checked. positions=positions are setting the cartesian coordinates.
            atoms = Atoms(numbers=atomic_numbers, positions=positions, cell=cell, pbc=[True, True, True])

            # I verified that the energy IS the energy that includes the correction (see curtis_read_alexandria.ipynb)
            calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
            atoms.set_calculator(calc)
            entries.append(atoms)

    print(f"found {num_configs} systems")
    print(f"after filtering, found {len(entries)} systems")
    return entries

if __name__ == "__main__":
    main()