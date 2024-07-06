import bz2
import numpy as np
import json
import os
import glob
from tqdm import tqdm
import pyarrow as pa # we are using pyarrow because of https://stackoverflow.com/questions/51361356/a-comparison-between-fastparquet-and-pyarrow
import pyarrow.parquet as pq
from pymatgen.core.periodic_table import Element

from abc import ABC, abstractmethod

batch_size = 1000
class DatasetStandardizer(ABC):
    def __init__(self, schema):
        self.schema = schema

    def prepare_parquet_file(self, raw_data_dir, output_dir):
        with pq.ParquetWriter(f"{output_dir}/dataset.parquet", self.schema) as writer:
            for data in self.data_generator(batch_size, raw_data_dir):
                table = pa.Table.from_pydict(data, schema=self.schema)

                # Write the table chunk to the Parquet file
                writer.write_table(table)

    @abstractmethod
    def data_generator(batch_size, raw_data_dir_path):
        pass

class AlexandriaStandardizer(DatasetStandardizer):
    def __init__(self):
        # NOTE: float64 is needed for the lattice. float32 is not enough.
        # e.g. This number cannot fit in a float32 so we need to use float64.
        # value = np.float32(6.23096541)
        # print(value)
        super().__init__(pa.schema([
            ('lattice', pa.list_(pa.list_(pa.float64(), 3), 3)),
            ('frac_coords', pa.list_(pa.list_(pa.float64(), 3))),
            ('atomic_numbers', pa.list_(pa.uint8())),
            ('energy', pa.float64()),
        ]))

    def data_generator(batch_size, raw_data_dir):
        # file_paths = sorted(glob.glob(f"{raw_dataset_input_dir}/*.json.bz2"))[4:] # use this to only process the last file
        file_paths = sorted(glob.glob(f"{raw_data_dir}/*.json.bz2"))

        for filepath in file_paths:
            with bz2.open(filepath, "rt", encoding="utf-8") as fh:
                data = json.load(fh)

                # Initialize lists to store each field
                lattices = []
                atomic_numbers = []
                frac_coords = []
                energies = []

                # Iterate through each entry
                for entry in tqdm(data["entries"]):
                    structure = entry["structure"]
                    lattices.append(structure["lattice"]["matrix"])
                    atomic_numbers.append([Element(site["label"]).Z for site in structure["sites"]])
                    frac_coords.append([site["abc"] for site in structure["sites"]])
                    energies.append(entry["energy"])

                yield {
                    "lattices": lattices,
                    "atomic_numbers": atomic_numbers,
                    "frac_coords": frac_coords,
                    "energies": energies,
                }