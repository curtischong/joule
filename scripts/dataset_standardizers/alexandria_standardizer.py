import bz2
import json
import glob
from tqdm import tqdm
from pymatgen.core.periodic_table import Element
import pyarrow as pa

from dataset_standardizer import DatasetStandardizer
from shared import data_dir


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

    def data_generator(self, raw_data_dir):
        file_paths = sorted(glob.glob(f"{raw_data_dir}/*.json.bz2"))[4:] # use this to only process the last file
        # file_paths = sorted(glob.glob(f"{raw_data_dir}/*.json.bz2"))
        assert len(file_paths) > 0, f"No files found in {raw_data_dir}"

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

def main():
    AlexandriaStandardizer().prepare_parquet_file(raw_data_dir=f"{data_dir}/raw/alexandria", output_dir=f"{data_dir}/standardized/alexandria")

if __name__ == "__main__":
    main()