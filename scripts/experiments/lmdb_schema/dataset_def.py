import numpy as np
from torch_geometric.data import Data

class DataDefField:
    def __init__(self, name, data, dtype, is_fixed_size=True):
        self.name = name
        self.data_bytes = data.tobytes()

        # the type of the data matters a lot since it affects how it's packed.
        # NOTE: we DO NOT want to do a type conversion here to "hotfix" if this assert fails, since it means the original datatype is wrong.
        assert data.dtype == dtype

        self.is_fixed_size = is_fixed_size

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
        self.num_atoms = packed_data[0: np.dtype(np.uint16).itemsize]

        res = Data()

        ptr = np.dtype(np.uint16).itemsize
        for field in self.fields:
            # TODO: this needs work
            if field.is_fixed_size:
                res[field.name] = field.from_bytes(packed_data[ptr: ptr + field.data_bytes.nbytes])
                ptr += field.data_bytes.nbytes
            else:
                field.from_bytes(packed_data[ptr:])
                ptr += field.data_bytes.nbytes
        return res



def main():
    # NOTE: float64 is needed for the lattice. float32 is not enough.
    # This number cannot fit in a float32 so we need to use float64.
    # value = np.float32(6.23096541)
    # print(value)

    lattice = np.array([[6.23096541, 0., 0.], [0., 6.23096541, 0.], [-3.1154827, -3.1154827, 6.28232566]])
    frac_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [0.5, 0.0, 0.0], [0.5, 0.5, 0.5]])
    atomic_numbers = np.array([30, 16, 16, 16, 49])
    energy = np.array([-16.5690])[0]

    datadef = DataDef(
        num_atoms=len(atomic_numbers),
        fields=[
        DataDefField("lattice", lattice, np.float64),
        DataDefField("frac_coords", frac_coords, np.float64),
        DataDefField("atomic_numbers", atomic_numbers, np.uint8), # range is: [0, 255]
        DataDefField("energy", energy, np.float64),
    ])

if __name__ == "__main__":
    main()