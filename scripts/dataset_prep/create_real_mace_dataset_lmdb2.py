from ase import Atoms
from fairchem.core.preprocessing import AtomsToGraphs
import ase.io
import lmdb
import pickle
from tqdm import tqdm
import torch
import os
import numpy as np
import time
import pymatgen
import json
import bz2
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.ase import AseAtomsAdaptor
from ase.calculators.singlepoint import SinglePointCalculator
import h5py
from multiprocessing import Process
import concurrent.futures

from dataset_prep_common import get_range, parse_config

IN_TRAIN_DIR = "datasets/real_mace/train"
IN_VAL_DIR = "datasets/real_mace/val"
OUT_TRAIN_DIR = "datasets/lmdb/real_mace/train"
OUT_VAL_DIR = "datasets/lmdb/real_mace/val"
# MAX_ATOMIC_NUMBER = 36
# MIN_ATOMIC_NUMBER = 36
# MAX_ATOMIC_NUMBER = 54

most_common_elements_only_one_per_sample = [8, 3, 15, 12, 16, 1, 25, 7, 26, 14, 9, 6, 29, 27, 11, 23, 19, 20, 13, 17] 
MAX_JOBS = 8


def main():
    config = parse_config()

    os.makedirs(OUT_TRAIN_DIR, exist_ok=True)
    os.makedirs(OUT_VAL_DIR, exist_ok=True)


    parse_datasets(config, IN_TRAIN_DIR, OUT_TRAIN_DIR, "train", num_files=64)
    parse_datasets(config, IN_VAL_DIR, OUT_VAL_DIR, "val", num_files=64)


def parse_datasets(config, in_dir, out_dir, in_dir_prefix, num_files):
    def process_file(i):
        entries = get_entries(in_dir, f"{in_dir_prefix}_{i}")
        db_name = f"{out_dir}/{i}"
        create_lmdb(config, db_name, entries)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_JOBS) as executor:
        futures = []
        for i in range(num_files):
            futures.append(executor.submit(process_file, i))
        
        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # to raise any exceptions occurred
            except Exception as e:
                print(f"An error occurred: {e}")


def create_lmdb(config, dataset_path, atoms: list[pymatgen.io.ase.MSONAtoms]):
    db = lmdb.open(
        f"{dataset_path}.lmdb",
        map_size=1099511627776 * 2, # two terabytes is the max size of the db
        subdir=False,
        meminit=False,
        map_async=True,
    )

    a2g = AtomsToGraphs(
        max_neigh=config["model"]["max_neighbors"],
        radius=config["model"]["cutoff"],
        r_energy=True,    # False for test data
        r_forces=True,    # False for test data
        r_distances=False,
        r_fixed=True,
    )

    start_time = time.time()

    tags = atoms[0].get_tags()
    data_objects = a2g.convert_all(atoms, disable_tqdm=True)

    print(f"reading {dataset_path}")


    for fid, data in tqdm(enumerate(data_objects), total=len(data_objects)):
        #assign sid
        data.sid = torch.LongTensor([0])

        #assign fid
        data.fid = torch.LongTensor([fid])

        #assign tags, if available
        data.tags = torch.LongTensor(tags)

        # Filter data if necessary
        # OCP filters adsorption energies > |10| eV and forces > |50| eV/A

        # no neighbor edge case check
        if data.edge_index.shape[1] == 0:
            print("no neighbors. skipping. fid=", fid)
            continue

        txn = db.begin(write=True)
        txn.put(f"{fid}".encode("ascii"), pickle.dumps(data, protocol=-1))
        txn.commit()

    txn = db.begin(write=True)
    txn.put(f"length".encode("ascii"), pickle.dumps(len(data_objects), protocol=-1))
    txn.commit()


    db.sync()
    db.close()

    end_time = time.time()
    print(f"{dataset_path} lmdb created")
    print(f"Time to create lmdb: {end_time - start_time}")

def get_entries(in_dir, file_name):
    entries = []

    with h5py.File(f"{in_dir}/{file_name}.h5", 'r') as hdf5_file:
        num_configs = len(hdf5_file["config_batch_0"])
        # num_configs = 1000
        for i in tqdm(range(num_configs)):
            config_group = hdf5_file[f'config_batch_0/config_{i}']
            atomic_numbers = config_group['atomic_numbers'][:]
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