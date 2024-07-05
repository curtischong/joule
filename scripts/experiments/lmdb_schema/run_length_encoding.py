import numpy as np

def encode_to_rle_bytes(byte_data):
    if len(byte_data) == 0:
        return bytes()

    # Convert byte_data to a NumPy array
    byte_array = np.frombuffer(byte_data, dtype=np.uint8)

    # Find the positions where the value changes
    pos = np.where(np.diff(byte_array) != 0)[0]
    # Add the end position
    pos = np.concatenate(([0], pos + 1, [len(byte_array)]))

    # Calculate the lengths of each run
    lengths = np.diff(pos)
    # Get the values of each run
    values = byte_array[pos[:-1]]

    # Convert values and lengths to bytes
    rle_byte_array = bytearray()
    for value, length in zip(values, lengths):
        rle_byte_array.append(value)
        rle_byte_array.append(length)
    return bytes(rle_byte_array)

def decode_from_rle_bytes(rle_byte_data):
    if len(rle_byte_data) == 0:
        return bytes()

    values = []
    lengths = []
    for i in range(0, len(rle_byte_data), 2):
        values.append(rle_byte_data[i])
        lengths.append(rle_byte_data[i + 1])

    # Reconstruct the original byte array
    decoded = bytearray()
    for value, length in zip(values, lengths):
        decoded.extend([value] * length)
    return bytes(decoded)

# Example usage
input_bytes = b'\x01\x01\x02\x02\x02\x03\x03\x01\x01\x01'
print("Original byte array:", input_bytes)

encoded_bytes = encode_to_rle_bytes(input_bytes)
print("Encoded to RLE bytes:", encoded_bytes)

decoded_bytes = decode_from_rle_bytes(encoded_bytes)
print("Decoded byte array:", decoded_bytes)