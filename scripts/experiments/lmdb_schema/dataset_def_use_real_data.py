from collections import Counter
import json
import numpy as np
from torch_geometric.data import Data

from enum import Enum
import brotli
import zlib
import lzma
import bz2
import time

from tqdm import tqdm
from run_length_encoding import encode_to_rle_bytes
from pymatgen.entries.computed_entries import ComputedStructureEntry

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

class DataDefField:
    def __init__(self, name: str, data: np.ndarray, dtype: np.dtype, data_shape: DataShape):
        self.name = name
        self.data_bytes = data.tobytes()

        # the type of the data matters a lot since it affects how it's packed.
        # NOTE: we DO NOT want to do a type conversion here to "hotfix" if this assert fails, since it means the original datatype is wrong.
        assert data.dtype == dtype
        self.dtype = dtype
        self.shape = data.shape

        self.data_shape = data_shape

class DataDef:
    def __init__(self, *, num_atoms: int, fields: list[DataDefField]):
        self.fields = fields

        self.packed_data = b""
        self.packed_data += np.uint16(num_atoms).tobytes() # use an unsigned short with range [0, 65535]
        for field in self.fields:
            self.packed_data += field.data_bytes
    
    def from_bytes(self, packed_data: bytes):
        res = Data()
        num_atoms = np.frombuffer(packed_data[0:np.dtype(np.uint16).itemsize], dtype=np.uint16)[0].item()

        ptr = np.dtype(np.uint16).itemsize
        for field in self.fields:
            data_len = self._data_len(field, num_atoms)
            res[field.name] = np.frombuffer(packed_data[ptr: ptr + data_len], dtype=field.dtype).reshape(field.data_shape.to_np_shape(num_atoms))
            ptr += data_len
        return res
    
    def _data_len(self, field: DataDefField, num_atoms: int):
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

def compress_and_read_entry(data: any, counter: Counter, ith_sample: int):
    entry = ComputedStructureEntry.from_dict(data["entries"][ith_sample]) # this ype conversion is slow. but I'm fine since we're not using this in prod

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
            # NOTE: float64 is needed for the lattice. float32 is not enough.
            # e.g. This number cannot fit in a float32 so we need to use float64.
            # value = np.float32(6.23096541)
            # print(value)
            DataDefField("lattice", lattice, np.float64, DataShape.MATRIX_3x3),
            DataDefField("frac_coords", frac_coords, np.float64, DataShape.MATRIX_nx3),
            DataDefField("atomic_numbers", atomic_numbers, np.uint8, DataShape.VECTOR), # range is: [0, 255]
            DataDefField("energy", energy, np.float64, DataShape.SCALAR),
        ])

    packed_data = datadef.packed_data

    time_start = time.time()

    brotli_compressed = brotli.compress(packed_data)
    zlib_compressed = zlib.compress(packed_data)
    pylzma_compressed = lzma.compress(packed_data)
    bz2_compressed = bz2.compress(packed_data)
    rle_compressed = encode_to_rle_bytes(packed_data)
    counter["uncompressed"] += len(packed_data)
    counter["brotli"] += len(brotli_compressed)
    counter["zlib"] += len(zlib_compressed)
    counter["pylzma"] += len(pylzma_compressed)
    counter["bz2"] += len(bz2_compressed)
    counter["rle"] += len(rle_compressed)
    # print(f"brotli compressed (len={len(brotli_compressed)})")
    # print(f"zlib compressed (len={len(zlib_compressed)})")
    # print(f"pylzma compressed (len={len(pylzma_compressed)})")
    # print(f"bz2 compressed (len={len(bz2_compressed)})")
    # print(f"rle compressed (len={len(rle_compressed)})")

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