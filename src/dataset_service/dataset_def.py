from typing import TypeAlias
import numpy as np
from torch_geometric.data import Data

from enum import Enum
import zlib

from abc import ABC, abstractmethod
from lmdb import Environment

from dataset_service.tools import int_to_bytes

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

    def read_entry(self, db: Environment, idx: int):
        compressed = db.begin().get(int_to_bytes(idx))
        return self._from_bytes(compressed)

    @abstractmethod
    def raw_data_to_lmdb(self, raw_dataset_input_dir: str, lmdb_output_dir: str):
        # please use tqdm to track progress
        pass

    def _pack_entry(self, entry_data: dict[FieldName, np.ndarray], num_atoms: int):
        packed_data = np.uint16(num_atoms).tobytes() # start off with the number of atoms

        for field in self.fields:
            field_data = entry_data[field.name]
            assert field_data.dtype == field.dtype
            packed_data += field_data.tobytes()

        return zlib.compress(packed_data) # we are using zlib since our experiemnt in scripts/experiments/lmdb_schema/dataset_def_use_real_data.py had the best results

    def _save_entry(self, db: Environment, data_idx: int, compressed: bytes):
        # TODO: investigate if we can be faster by not committing every time
        # https://github.com/jnwatson/py-lmdb/issues/63
        # I think commiting everytime is slightly better since it automates pointer incrementation (and it's done in c, not python)
        txn = db.begin(write=True)
        txn.put(int_to_bytes(data_idx), compressed)
        txn.commit()
    
    def _from_bytes(self, packed_data: bytes):
        res = Data()
        num_atoms = np.frombuffer(packed_data[0:np.dtype(np.uint16).itemsize], dtype=np.uint16)[0].item()

        ptr = np.dtype(np.uint16).itemsize
        for field in self.fields:
            data_len = self._data_len(field, num_atoms)
            compressed_data = packed_data[ptr: ptr + data_len]
            uncompressed_data = zlib.decompress(compressed_data)
            res[field.name] = np.frombuffer(uncompressed_data, dtype=field.dtype).reshape(field.data_shape.to_np_shape(num_atoms))
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
    