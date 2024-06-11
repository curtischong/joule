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

from dataset_prep_common import get_range, parse_config

DATASET_DIR = "datasets/lmdb"

def main():
    config = parse_config()

    os.makedirs(DATASET_DIR, exist_ok=True)

    create_dataset(config, "1")
    create_dataset(config, "10")
    create_dataset(config, "1000")
    # create_dataset(config, "10000")
    # create_dataset(config, "all") # Too slow rn


def create_dataset(config, dataset_type: str):
    entries = get_entries()
    np.random.shuffle(entries)
    n = len(entries)

    print(f"found {n} systems")
    ranges = get_range(n, dataset_type)

    train_entries = entries[ranges[0][0]:ranges[0][1]]
    val_entries = entries[ranges[1][0]:ranges[1][1]]
    test_entries = entries[ranges[2][0]:ranges[2][1]]
    print(f"train len: {len(train_entries)}, val len: {len(val_entries)}, test len: {len(test_entries)}")

    db_name = f"{DATASET_DIR}/alexandria_{dataset_type}"
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
    idx = 0

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

def read_trajectory_extract_features(a2g, traj_path: str):
    traj = ase.io.read(traj_path, ":")
    tags = traj[0].get_tags()
    images = [traj[0], traj[-1]]
    data_objects = a2g.convert_all(images, disable_tqdm=True)
    data_objects[0].tags = torch.LongTensor(tags)
    data_objects[1].tags = torch.LongTensor(tags)
    return data_objects

def get_entries():
    IN_DIR = f"/home/ubuntu/joule/datasets/alexandria"
    filename = "alexandria_ps_004"
    with bz2.open(f"{IN_DIR}/{filename}.json.bz2", "rt", encoding="utf-8") as fh:
        data = json.load(fh)
        entries = []
        for i in data["entries"]:
            computed_entry = ComputedStructureEntry.from_dict(i)
            structure = computed_entry.structure
            atoms = AseAtomsAdaptor.get_atoms(structure)
            # forces = computed_entry.data.get('forces', None)  # Replace with actual forces if available
            forces = []
            for site in structure:
                if "forces" in site.properties:
                    forces.append(site.properties["forces"])
                else:
                    forces.append([None, None, None])  # If forces are not present for a site
            # print(forces)
            # forces = np.array(atoms.get_forces())
            uncorrected_energy = computed_entry.energy
            correction = computed_entry.correction
            corrected_energy = uncorrected_energy + correction

            calc = SinglePointCalculator(atoms, energy=corrected_energy, forces=forces)
            atoms.set_calculator(calc)
            entries.append(atoms)
    return entries

if __name__ == "__main__":
    main()