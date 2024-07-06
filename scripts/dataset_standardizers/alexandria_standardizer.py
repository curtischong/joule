import bz2
import numpy as np
import json
import os
import glob
from tqdm import tqdm
import pyarrow as pa # we are using pyarrow because of https://stackoverflow.com/questions/51361356/a-comparison-between-fastparquet-and-pyarrow
import pyarrow.parquet as pq
from pymatgen.core.periodic_table import Element

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
        
    def raw_data_to_lmdb(self, raw_dataset_input_dir: str, lmdb_output_dir: str, max_entries_per_db = 1000000):
        os.makedirs(lmdb_output_dir, exist_ok=True)

        db = None
        # file_paths = sorted(glob.glob(f"{raw_dataset_input_dir}/*.json.bz2"))[4:] # use this to only process the last file
        file_paths = sorted(glob.glob(f"{raw_dataset_input_dir}/*.json.bz2"))
        assert len(file_paths) > 0, f"No files found in {raw_dataset_input_dir}"


        for filepath in file_paths:
            with bz2.open(filepath, "rt", encoding="utf-8") as fh:
                data = json.load(fh)
                print(f"processing {filepath}")
                for idx_in_file in tqdm(range(len(data["entries"]))):
                    db = self._open_write_db(lmdb_output_dir, ith_db)

                    entry = data["entries"][idx_in_file]
                    self._parse_entry_and_save(db, entry, idx_in_db)

        # Create a PyArrow Table
        table = pa.Table.from_arrays([pa_array1, pa_array2], names=['array1', 'array2'])

        db.sync()
        db.close()

    def data_generator(batch_size, file_paths):
        for filepath in file_paths:
            with bz2.open(filepath, "rt", encoding="utf-8") as fh:
                data = json.load(fh)

                num_entries = len(data["entries"])
                max_num_sites = max(len(entry["structure"]["sites"]) for entry in data["entries"])

                # Initialize empty NumPy arrays with the appropriate shapes
                lattices = np.empty((num_entries, 3, 3), dtype=np.float64)
                atomic_numbers = np.empty((num_entries, max_num_sites), dtype=np.uint8)
                frac_coords = np.empty((num_entries, max_num_sites, 3), dtype=np.float64)
                energies = np.empty(num_entries, dtype=np.float64)

                for i, entry in enumerate(tqdm(data["entries"])):
                    structure = entry["structure"]
                    
                    lattices[i] = np.array(structure["lattice"]["matrix"], dtype=np.float64)
                    
                    num_sites = len(structure["sites"])
                    atomic_numbers[i, :num_sites] = np.array([Element(site["label"]).Z for site in structure["sites"]], dtype=np.uint8)
                    frac_coords[i, :num_sites, :] = np.array([site["abc"] for site in structure["sites"]], dtype=np.float64)
                    
                    # If there are fewer sites than the maximum, fill the rest with a placeholder, e.g., 0
                    if num_sites < max_num_sites:
                        atomic_numbers[i, num_sites:] = 0
                        frac_coords[i, num_sites:, :] = 0.0
                    
                    energies[i] = np.float64(entry["energy"])
                yield entry_data