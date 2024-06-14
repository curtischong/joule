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

from dataset_prep_common import get_range, parse_config

IN_TRAIN_DIR = "datasets/real_mace/train"
IN_VAL_DIR = "datasets/real_mace/val"
TRAIN_DIR = "datasets/lmdb/real_mace/train"
VAL_DIR = "datasets/lmdb/real_mace/val"
# MAX_ATOMIC_NUMBER = 36
MAX_ATOMIC_NUMBER = 54


def main():
    config = parse_config()

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    num_train_files = 64
    for i in range(num_train_files):
        entries = get_entries(IN_TRAIN_DIR, f"train_{i}")
        db_name = f"{TRAIN_DIR}/{i}"
        create_lmdb(config, db_name, entries)

    num_val_files = 64
    for i in range(num_val_files):
        entries = get_entries(IN_VAL_DIR, f"val_{i}")
        db_name = f"{VAL_DIR}/{i}"
        create_lmdb(config, db_name, entries)

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
        for i in tqdm(range(num_configs)):
            config_group = hdf5_file[f'config_batch_0/config_{i}']
            atomic_numbers = config_group['atomic_numbers'][:]
            if not all([element <= MAX_ATOMIC_NUMBER for element in atomic_numbers]):
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