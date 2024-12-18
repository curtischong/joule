from fairchem.core.common.utils import build_config, load_config
from fairchem.core.preprocessing import AtomsToGraphs
from fairchem.core.datasets import LmdbDataset
import ase.io
from ase.build import bulk
from ase.build import fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS
import matplotlib.pyplot as plt
import lmdb
import pickle
from tqdm import tqdm
import torch
import os
import glob
import numpy as np
from fairchem.core.common.flags import flags
import argparse
import time

from dataset_prep_common import get_range, parse_config

DATASET_DIR = "datasets/lmdb"

def main():
    config = parse_config()

    os.makedirs(DATASET_DIR, exist_ok=True)

    create_dataset(config, "1")
    create_dataset(config, "10")
    create_dataset(config, "1000")
    create_dataset(config, "10000")
    # create_dataset(config, "all") # Too slow rn


def create_dataset(config, dataset_type: str):
    data_paths = get_traj_files("datasets/mptrj-gga-ggapu/mptrj-gga-ggapu")
    np.random.shuffle(data_paths)
    n = len(data_paths)

    print(f"found {n} systems")
    ranges = get_range(n, dataset_type)

    train_paths = data_paths[ranges[0][0]:ranges[0][1]]
    val_paths = data_paths[ranges[1][0]:ranges[1][1]]
    test_paths = data_paths[ranges[2][0]:ranges[2][1]]
    print(f"train len: {len(train_paths)}, val len: {len(val_paths)}, test len: {len(test_paths)}")

    db_name = f"{DATASET_DIR}/mace_{dataset_type}"
    create_lmdb(config, f"{db_name}_val", val_paths)
    create_lmdb(config, f"{db_name}_test", test_paths)
    create_lmdb(config, f"{db_name}_train", train_paths) # train last since it'll be the slowest


def create_lmdb(config, dataset_path, data_paths: list[str]):
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
    idx = 0

    start_time = time.time()
    for system in data_paths:
        # Extract Data object
        data_objects = read_trajectory_extract_features(a2g, system)
        initial_struc = data_objects[0]
        relaxed_struc = data_objects[1]

        initial_struc.y_init = initial_struc.y # subtract off reference energy, if applicable
        del initial_struc.y
        initial_struc.y_relaxed = relaxed_struc.y # subtract off reference energy, if applicable
        initial_struc.pos_relaxed = relaxed_struc.pos

        # Filter data if necessary
        # FAIRChem filters adsorption energies > |10| eV

        initial_struc.sid = idx  # arbitrary unique identifier

        # no neighbor edge case check
        if initial_struc.edge_index.shape[1] == 0:
            print("no neighbors", system)
            continue

        # Write to LMDB
        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(initial_struc, protocol=-1))
        txn.commit()
        db.sync()
        idx += 1

    db.close()
    end_time = time.time()
    print(f"{dataset_path} lmdb created")
    print(f"Time to create lmdb: {end_time - start_time}")

def read_trajectory_extract_features(a2g, traj_path: str):
    traj = ase.io.read(traj_path, ":")
    tags = traj[0].get_tags()
    images = [traj[0], traj[-1]]
    data_objects = a2g.convert_all(images, disable_tqdm=True)
    data_objects[0].tags = torch.LongTensor(tags)
    data_objects[1].tags = torch.LongTensor(tags)
    return data_objects

def get_traj_files(directory_path:str):
    pattern = os.path.join(directory_path, "*.extxyz")
    return glob.glob(pattern)

if __name__ == "__main__":
    main()