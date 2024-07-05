import lmdb
from dataset_service.dataset_def import DatasetDef, DataShape, FieldDef
import numpy as np
from tqdm import tqdm
import bz2
import json
import glob
from pymatgen.core.periodic_table import Element
import os

class AlexandriaDataset(DatasetDef):
    def __init__(self):
        super().__init__(
            fields=[
                # NOTE: float64 is needed for the lattice. float32 is not enough.
                # e.g. This number cannot fit in a float32 so we need to use float64.
                # value = np.float32(6.23096541)
                # print(value)
                FieldDef("lattice", np.float64, DataShape.MATRIX_3x3),
                FieldDef("frac_coords", np.float64, DataShape.MATRIX_nx3),
                FieldDef("atomic_numbers", np.uint8, DataShape.VECTOR), # range is: [0, 255]
                FieldDef("energy", np.float64, DataShape.SCALAR),
            ])
        
    def raw_data_to_lmdb(self, dataset_dir: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        db = lmdb.open(
            f"{output_dir}.lmdb", # TODO out dir
            map_size=1099511627776 * 2, # two terabytes is the max size of the db
            subdir=False,
            meminit=False,
            map_async=True,
        )
        for filepath in tqdm(glob.glob(f"{dataset_dir}/*.json.bz2")):
            with bz2.open(filepath, "rt", encoding="utf-8") as fh:
                data = json.load(fh)
                print(f"processing {filepath}")
                for i in tqdm(range(len(data["entries"]))):
                    entry = data["entries"][i]
                    structure = entry["structure"]

                    entry_data = {
                        "lattice": structure["lattice"]["matrix"],
                        "atomic_numbers": [Element(site["label"]).Z for site in structure["sites"]],
                        "frac_coords": [site["abc"] for site in structure["sites"]],
                        "energy": entry["energy"],
                    }
                    compressed = self._pack_entry(db, entry_data, len(structure["sites"]))
                    self._save_entry(db, i, compressed)

        db.sync()
        db.close()
    
    def read_lmdb(self, db_path: str):
        db = lmdb.open(
            db_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        for i in tqdm(range(db.stat()["entries"])):
            compressed = db.get(self._int_to_bytes(i))
            if compressed is None:
                continue
            entry = self.from_bytes(compressed)
            yield entry