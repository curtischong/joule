import numpy as np
from torch_geometric.data import Data

from enum import Enum

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
        self.num_atoms = num_atoms
        self.fields = fields

    def to_bytes(self):
        packed_data = b""
        packed_data += np.uint16(self.num_atoms).tobytes() # use an unsigned short with range [0, 65535]
        for field in self.fields:
            packed_data += field.data_bytes
        return packed_data
    
    def from_bytes(self, packed_data: bytes):
        res = Data()

        ptr = np.dtype(np.uint16).itemsize
        for field in self.fields:
            data_len = self._data_len(field)
            res[field.name] = np.frombuffer(packed_data[ptr: ptr + data_len], dtype=field.dtype).reshape(field.data_shape.to_np_shape(self.num_atoms))
            ptr += data_len
        return res
    
    def _data_len(self, field: DataDefField):
        data_shape = field.data_shape
        match data_shape:
            case DataShape.SCALAR:
                return np.dtype(field.dtype).itemsize
            case DataShape.VECTOR:
                return np.dtype(field.dtype).itemsize * self.num_atoms
            case DataShape.MATRIX_3x3:
                return np.dtype(field.dtype).itemsize * 9
            case DataShape.MATRIX_nx3:
                return np.dtype(field.dtype).itemsize * self.num_atoms * 3



def main():
    # NOTE: float64 is needed for the lattice. float32 is not enough.
    # This number cannot fit in a float32 so we need to use float64.
    # value = np.float32(6.23096541)
    # print(value)

    lattice = np.array([[6.23096541, 0., 0.], [0., 6.23096541, 0.], [-3.1154827, -3.1154827, 6.28232566]], dtype=np.float64)
    frac_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [0.5, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=np.float64)
    atomic_numbers = np.array([30, 16, 16, 16, 49], dtype=np.uint8)
    energy = np.array([-16.5690], dtype=np.float64)[0]

    datadef = DataDef(
        num_atoms=len(atomic_numbers),
        fields=[
        DataDefField("lattice", lattice, np.float64, DataShape.MATRIX_3x3),
        DataDefField("frac_coords", frac_coords, np.float64, DataShape.MATRIX_nx3),
        DataDefField("atomic_numbers", atomic_numbers, np.uint8, DataShape.VECTOR), # range is: [0, 255]
        DataDefField("energy", energy, np.float64, DataShape.SCALAR),
    ])

    b = datadef.to_bytes()
    parsed_data = datadef.from_bytes(b)
    for key, value in parsed_data.items():
        print(key, value)
    print(parsed_data["energy"])

if __name__ == "__main__":
    main()