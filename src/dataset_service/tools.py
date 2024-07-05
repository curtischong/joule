
# store the keys as bytes into the LMDB for smaller keys
def int_to_bytes(x: int):
    return x.to_bytes((x.bit_length() + 7) // 8, 'big') or b'\0'

# Function to convert bytes back to integer
def bytes_to_int(b: bytes):
    return int.from_bytes(b, 'big')
