import struct
import zlib
import brotli
import numpy as np

data = {
    'lattice': np.array([[6.23096541, 0., 0.], [0., 6.23096541, 0.], [-3.1154827, -3.1154827, 6.28232566]]),
    'frac_coords': np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [0.5, 0.0, 0.0], [0.5, 0.5, 0.5]]),
    'atomic_numbers': np.array([30, 16, 16, 16, 49]),
    'energy': np.array([-16.5690])[0],
}

print(data["energy"].tobytes())

keys = list(data.keys())
num_atoms = len(data["atomic_numbers"])

packed_data = b""
packed_data += np.uint16(num_atoms).tobytes() # use an unsigned short with range [0, 65535]

# TODO: wrapper class that determines if each field is a fixed-size array or a variable-sized array
# when we WRITE the code to pack the dataset's data, we need to specify if the variable is fixed-size or variable-size

for key in keys:
    packed_data += data[key].tobytes()

# Note: to interpret the bytes printed to the console, see this: https://claude.ai/chat/7f68e8c4-51ca-4553-903a-f21ea513a5df
# print(f"Packed Data (len={len(packed_data)}): {packed_data}")
# print(f"brotli compressed (len={len(brotli.compress(packed_data))}): {brotli.compress(packed_data)}")
# print(f"zlib compressed (len={len(zlib.compress(packed_data))}): {zlib.compress(packed_data)}")


# Unpack the data from the bytes object
# unpacked_data = struct.unpack(format_string, packed_data)

# # Create a new dictionary from the unpacked data
# unpacked_dict = {
#     'id': unpacked_data[0],
#     'value': unpacked_data[1],
#     'active': unpacked_data[2]
# }

# print(f"Unpacked Data: {unpacked_dict}")