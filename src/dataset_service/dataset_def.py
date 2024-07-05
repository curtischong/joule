from collections import Counter
import json
from typing import TypeAlias
import numpy as np
from torch_geometric.data import Data

from enum import Enum
import brotli
import zlib
import lzma
import bz2
import time
import zlib

from tqdm import tqdm
from pymatgen.entries.computed_entries import ComputedStructureEntry
from abc import ABC, abstractmethod
from lmdb import Environment

class DataShape(Enum):
    SCALAR = 0
    VECTOR = 1 # 1D tensor
    MATRIX_3x3 = 2
    MATRIX_nx3 = 3

    def to_np_shape(self, num_atoms: int):
        match self:
            case DataShape.SCALAR:
                return 1,
            case DataShape.VECTOR:
                return (num_atoms,)
            case DataShape.MATRIX_3x3:
                return (3, 3)
            case DataShape.MATRIX_nx3:
                return (num_atoms, 3)

FieldName: TypeAlias = str
class FieldDef:
    def __init__(self, name: FieldName, dtype: np.dtype, data_shape: DataShape):
        self.name = name
        self.dtype = dtype
        self.data_shape = data_shape

class DatasetDef(ABC):
    def __init__(self, fields: list[FieldDef]):
        self.fields = fields

    # store the keys as bytes into the LMDB for smaller keys
    def _int_to_bytes(self, x: int):
        return x.to_bytes((x.bit_length() + 7) // 8, 'big') or b'\0'
    
    # Function to convert bytes back to integer
    def _bytes_to_int(self, b: bytes):
        return int.from_bytes(b, 'big')
    
    def _pack_entry(self, entry_data: dict[FieldName, np.ndarray], num_atoms: int):
        packed_data = np.uint16(num_atoms).tobytes() # start off with the number of atoms

        for field in self.fields:
            packed_data += entry_data[field.name].tobytes()

        return zlib.compress(packed_data) # we are using zlib since our experiemnt in scripts/experiments/lmdb_schema/dataset_def_use_real_data.py had the best results

    def _save_entry(self, db: Environment, data_idx: int, compressed: bytes):
        # TODO: investigate if we can be faster by not committing every time
        # https://github.com/jnwatson/py-lmdb/issues/63
        # I think commiting everytime is slightly better since it automates pointer incrementation (and it's done in c, not python)
        txn = db.begin(write=True)
        txn.put(self._int_to_bytes(data_idx), compressed)
        txn.commit()
    
    def _from_bytes(self, packed_data: bytes):
        res = Data()
        num_atoms = np.frombuffer(packed_data[0:np.dtype(np.uint16).itemsize], dtype=np.uint16)[0].item()

        ptr = np.dtype(np.uint16).itemsize
        for field in self.fields:
            data_len = self._data_len(field, num_atoms)
            res[field.name] = np.frombuffer(packed_data[ptr: ptr + data_len], dtype=field.dtype).reshape(field.data_shape.to_np_shape(num_atoms))
            ptr += data_len
        return res
    
    def _data_len(self, field: FieldDef, num_atoms: int):
        data_shape = field.data_shape
        match data_shape:
            case DataShape.SCALAR:
                return np.dtype(field.dtype).itemsize
            case DataShape.VECTOR:
                return np.dtype(field.dtype).itemsize * num_atoms
            case DataShape.MATRIX_3x3:
                return np.dtype(field.dtype).itemsize * 9
            case DataShape.MATRIX_nx3:
                return np.dtype(field.dtype).itemsize * num_atoms * 3
    
    @abstractmethod
    def raw_data_to_lmdb(self, dataset_dir: str):
        # please use tqdm to track progress
        pass


def compress_and_read_entry(data: any, counter: Counter, ith_sample: int):
    entry = ComputedStructureEntry.from_dict(data["entries"][ith_sample])

    structure = entry.structure
    atomic_numbers = np.array([site.specie.number for site in structure], dtype=np.uint8)
    lattice = structure.lattice.matrix
    frac_coords = structure.frac_coords
    energy = np.float64(entry.energy)
    # print(f"atomic_numbers: {atomic_numbers}")
    # print(f"lattice: {lattice}")
    # print(f"frac_coords: {frac_coords}")
    # print(f"Energy: {energy}")

    datadef = DataDef(
        num_atoms=len(atomic_numbers),
        fields=[
            FieldDef("lattice", lattice, np.float64, DataShape.MATRIX_3x3),
            FieldDef("frac_coords", frac_coords, np.float64, DataShape.MATRIX_nx3),
            FieldDef("atomic_numbers", atomic_numbers, np.uint8, DataShape.VECTOR), # range is: [0, 255]
            FieldDef("energy", energy, np.float64, DataShape.SCALAR),
        ])

    packed_data = datadef.packed_data

    time_start = time.time()

    brotli_compressed = brotli.compress(packed_data)
    zlib_compressed = zlib.compress(packed_data)
    pylzma_compressed = lzma.compress(packed_data)
    bz2_compressed = bz2.compress(packed_data)
    counter["uncompressed"] += len(packed_data)
    counter["brotli"] += len(brotli_compressed)
    counter["zlib"] += len(zlib_compressed)
    counter["pylzma"] += len(pylzma_compressed)
    counter["bz2"] += len(bz2_compressed)
    # print(f"brotli compressed (len={len(brotli_compressed)})")
    # print(f"zlib compressed (len={len(zlib_compressed)})")
    # print(f"pylzma compressed (len={len(pylzma_compressed)})")
    # print(f"bz2 compressed (len={len(bz2_compressed)})")

    # print(f"time took to compress: {time.time() - time_start}")


    parsed_data = datadef.from_bytes(packed_data)
    assert np.array_equal(lattice, parsed_data["lattice"])
    assert np.array_equal(frac_coords, parsed_data["frac_coords"])
    assert energy == parsed_data["energy"]
    assert np.array_equal(atomic_numbers, parsed_data["atomic_numbers"])

def main():
    IN_DIR = "../../../datasets/alexandria"
    filename = "alexandria_ps_004"
    with bz2.open(f"{IN_DIR}/{filename}.json.bz2", "rt", encoding="utf-8") as fh:
        data = json.load(fh)

    # compress_and_read_entry(data, 1)
    c = Counter()
    for i in tqdm(range(len(data["entries"]))):
        compress_and_read_entry(data, c, i)
    print(f"num_entries: {len(data['entries'])}")
    for k, v in c.items():
        print(f"{k:12} {v}")

    """
    num_entries: 15419
    brotli       13960
    zlib         10814
    pylzma       16852
    bz2          17513
    rle          22392
    """

if __name__ == "__main__":
    main()