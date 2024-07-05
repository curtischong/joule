from dataset_service.dataset_def import DatasetDef, DataShape, FieldDef
import numpy as np
import tqdm
import os
import bz2
import json

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
        
    def raw_data_to_lmdb(self, dataset_dir: str):
        for filename in tqdm(os.listdir(dataset_dir)):
            with bz2.open(f"{dataset_dir}/{filename}.json.bz2", "rt", encoding="utf-8") as fh:
                data = json.load(fh)
                print(f"processing {filename}")
                for i in tqdm(range(len(data["entries"]))):
                    entry = data["entries"][i]
                    # compress_and_read_entry(data, c, i)
                    pass