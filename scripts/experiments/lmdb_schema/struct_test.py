import struct
import zlib
import brotli

# Define the format string
format_string = 'if?'

data = {
    'id': 1,
    'value': 3.14,
    'active': True
}

# Pack the data into a bytes object
packed_data = struct.pack(format_string, data['id'], data['value'], data['active'])

# TODO: we need to compress the data before storing it in lmdb
# print(struct.pack(format_string, data['value']))
print(f"Packed Data: {packed_data}")
print(f"compressed: {zlib.compress(packed_data)}")


# Unpack the data from the bytes object
unpacked_data = struct.unpack(format_string, packed_data)

# Create a new dictionary from the unpacked data
unpacked_dict = {
    'id': unpacked_data[0],
    'value': unpacked_data[1],
    'active': unpacked_data[2]
}

print(f"Unpacked Data: {unpacked_dict}")