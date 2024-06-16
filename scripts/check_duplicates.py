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

IN_TRAIN_DIR = "datasets/real_mace/train"
IN_VAL_DIR = "datasets/real_mace/val"
OUT_TRAIN_DIR = "datasets/lmdb/real_mace3/train"
OUT_VAL_DIR = "datasets/lmdb/real_mace3/val"

MAX_JOBS = 8

hashed = {} # hash -> idx of the entry

"""
duplicate found at 9000 and 3666
[[1.17386892 0.71884384 3.89330044]
 [0.59527832 0.36452604 1.97431853]
 [1.75246006 1.07315113 5.81228413]][3 8 8]Cell([[2.30281076, -0.39696027, 2.29894373], [0.69315251, 2.23160376, 2.29894368], [-0.648
23271, -0.39696027, 3.18871066]])
"""

def main():
    os.makedirs(OUT_TRAIN_DIR, exist_ok=True)
    os.makedirs(OUT_VAL_DIR, exist_ok=True)


    results = parse_datasets(IN_TRAIN_DIR, "train", num_files=1) # TODO: increase num_files
    # results.extend(parse_datasets(IN_VAL_DIR, "val", num_files=64))

    for i, res in enumerate(results):
        hash = str(res.get_positions()) + str(res.get_atomic_numbers()) + str(res.get_cell())
        if hash in hashed:
            print(f"duplicate found at {i} and {hashed[hash]}")
            print(hash)
            return
        hashed[hash] = i
    print("all done")

def parse_datasets(in_dir, in_dir_prefix, num_files):
    results = []
    for i in range(num_files):
        print(f"parsing {in_dir_prefix}_{i}")
        results.extend(get_entries(in_dir, f"{in_dir_prefix}_{i}"))
    return results


# this should be a list of pymatgen.io.ase.MSONAtoms
def create_lmdb(dataset_path, atoms: list[any]):
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
            # tags=tags,
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
            # if not all([element in most_common_elements_only_one_per_sample for element in atomic_numbers]):
            #     continue


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