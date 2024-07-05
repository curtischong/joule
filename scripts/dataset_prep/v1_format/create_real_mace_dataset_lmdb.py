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

DATASET_DIR = "datasets/lmdb/real_mace"
# MAX_ATOMIC_NUMBER = 36
MAX_ATOMIC_NUMBER = 54


def main():
    config = parse_config()

    os.makedirs(DATASET_DIR, exist_ok=True)

    entries = get_entries(3)
    np.random.shuffle(entries)

    create_dataset(entries, config, "1")
    create_dataset(entries, config, "10")
    create_dataset(entries, config, "1000")
    # create_dataset(entries, config, "10000")
    create_dataset(entries, config, "all") # Too slow rn

def read_data(file_path, num_configs=25):
    properties = {
        "atomic_numbers": [],
        "cell": [],
        "charges": [],
        "energy": [],
        "forces": [],
        "positions": [],
        "stress": [],
        "virials": [],
    }

    with h5py.File(file_path, 'r') as hdf5_file:
        for i in range(num_configs):
            config_group = hdf5_file[f'config_batch_0/config_{i}']
            properties["atomic_numbers"].append(config_group['atomic_numbers'][:])
            properties["cell"].append(config_group['cell'][:])
            properties["charges"].append(config_group['charges'][:])
            properties["energy"].append(config_group['energy'][()])
            properties["forces"].append(config_group['forces'][:])
            properties["positions"].append(config_group['positions'][:])
            properties["stress"].append(config_group['stress'][:])
            properties["virials"].append(config_group['virials'][:])

    return properties

def create_dataset(entries, config, dataset_type: str):
    n = len(entries)
    ranges = get_range(n, dataset_type)

    train_entries = entries[ranges[0][0]:ranges[0][1]]
    val_entries = entries[ranges[1][0]:ranges[1][1]]
    test_entries = entries[ranges[2][0]:ranges[2][1]]
    print(f"train len: {len(train_entries)}, val len: {len(val_entries)}, test len: {len(test_entries)}")

    db_name = f"{DATASET_DIR}/{dataset_type}"
    create_lmdb(config, f"{db_name}_val", val_entries)
    create_lmdb(config, f"{db_name}_test", test_entries)
    create_lmdb(config, f"{db_name}_train", train_entries) # train last since it'll be the slowest


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

def get_entries(file_idx):
    IN_DIR = "./datasets/real_mace"
    entries = []

    with h5py.File(f"{IN_DIR}/train_{file_idx}.h5", 'r') as hdf5_file:
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