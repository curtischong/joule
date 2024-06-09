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

def main():
    a2g = AtomsToGraphs(
        max_neigh=50,
        radius=6,
        r_energy=True,    # False for test data
        r_forces=True,    # False for test data
        r_distances=False,
        r_fixed=True,
    )
    db = lmdb.open(
        "sample_CuCO.lmdb",
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    traj_path = "CuCO_adslab.traj"

    system_paths = []
    idx = 0

    for system in system_paths:
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
            print("no neighbors", traj_path)
            continue

        # Write to LMDB
        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(initial_struc, protocol=-1))
        txn.commit()
        db.sync()
        idx += 1

    db.close()

def read_trajectory_extract_features(a2g, traj_path):
    traj = ase.io.read(traj_path, ":")
    tags = traj[0].get_tags()
    images = [traj[0], traj[-1]]
    data_objects = a2g.convert_all(images, disable_tqdm=True)
    data_objects[0].tags = torch.LongTensor(tags)
    data_objects[1].tags = torch.LongTensor(tags)
    return data_objects

if __name__ == "main":
    main()