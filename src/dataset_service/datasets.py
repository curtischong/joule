import lmdb
from dataset_service.dataset_def import DatasetDef, DataShape, FieldDef
import numpy as np
from tqdm import tqdm
import bz2
import json
import glob
from pymatgen.core.periodic_table import Element
import os

# This class handles marhalling and unmarshalling of the data
# the reason why it does NOT have its own db connection inside is because
# we do NOT know if you are writing to the db or reading from it (which requires diff .open() settings)
class AlexandriaDataset(DatasetDef):
    def __init__(self):
        super().__init__([
                # NOTE: float64 is needed for the lattice. float32 is not enough.
                # e.g. This number cannot fit in a float32 so we need to use float64.
                # value = np.float32(6.23096541)
                # print(value)
                FieldDef("lattice", np.float64, DataShape.MATRIX_3x3),
                FieldDef("frac_coords", np.float64, DataShape.MATRIX_nx3),
                FieldDef("atomic_numbers", np.uint8, DataShape.VECTOR), # range is: [0, 255]
                FieldDef("energy", np.float64, DataShape.SCALAR),
            ])
        
    def raw_data_to_lmdb(self, raw_dataset_input_dir: str, lmdb_output_dir: str, max_entries_per_db = 10000):
        os.makedirs(lmdb_output_dir, exist_ok=True)

        ith_total_entry = 0
        db = None
        file_paths = sorted(glob.glob(f"{raw_dataset_input_dir}/*.json.bz2"))[4:]
        assert len(file_paths) > 0, f"No files found in {raw_dataset_input_dir}"


        for filepath in file_paths:
            with bz2.open(filepath, "rt", encoding="utf-8") as fh:
                data = json.load(fh)
                print(f"processing {filepath}")
                for idx_in_file in tqdm(range(len(data["entries"]))):
                    idx_in_db = ith_total_entry % max_entries_per_db
                    if idx_in_db == 0:
                        if db is not None:
                            db.sync()
                            db.close()

                        ith_db = ith_total_entry // max_entries_per_db
                        db = self._open_db(lmdb_output_dir, ith_db)

                    entry = data["entries"][idx_in_file]
                    self._parse_entry_and_save(db, entry, idx_in_db)

                    ith_total_entry += 1

        db.sync()
        db.close()

    def _open_db(self, lmdb_output_dir: str, ith_db: int = 0):
        return lmdb.open(
            f"{lmdb_output_dir}/{ith_db}.lmdb", # TODO out dir
            map_size=1099511627776 * 2, # two terabytes is the max size of the db
            subdir=False,
            meminit=False,
            map_async=True,
        )

    def _parse_entry_and_save(self, db: lmdb.Environment, entry: any, idx_in_db: int):
        structure = entry["structure"]

        entry_data = {
            "lattice": np.array(structure["lattice"]["matrix"], dtype=np.float64),
            "atomic_numbers": np.array([Element(site["label"]).Z for site in structure["sites"]], dtype=np.uint8),
            "frac_coords": np.array([site["abc"] for site in structure["sites"]], dtype=np.float64),
            "energy": np.float64(entry["energy"]),
        }
        compressed = self._pack_entry(entry_data, len(structure["sites"]))
        self._save_entry(db, idx_in_db, compressed)