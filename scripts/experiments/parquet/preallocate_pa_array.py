import time
import pyarrow as pa
import numpy as np
import random

# def use_builder_to_create_array(length):
#     # Initialize the builder for int32 type
#     # builder = pa.int32()
#     builder = pa.array_builder(pa.int32())
#     for i in range(length):
#         builder.append(i)
    
#     # Finalize the builder to create an array
#     array = builder.finish()
#     return array

def use_numpy_to_create_array(length):
    numpy_array = np.empty(length, dtype=np.int32)
    for i in range(length):
        # numpy_array[i] = random.randint(0, 100)
        numpy_array[i] = i

    return pa.array(numpy_array)

def use_list_to_create_array(length):
    list_of_ints = []
    for i in range(length):
        # list_of_ints.append(random.randint(0, 100))
        list_of_ints.append(i)
    return pa.array(list_of_ints)


def main():
    length = 10000000
    # start_time = time.time()
    # array = use_builder_to_create_array(length)
    # end_time = time.time()
    # print(f"Time taken to preallocate list of length {length}: {end_time - start_time} seconds")

    start_time = time.time()
    array = use_numpy_to_create_array(length)
    end_time = time.time()
    print(f"Time taken to preallocate numpy array of length {length}: {end_time - start_time} seconds")

    start_time = time.time()
    array = use_list_to_create_array(length)
    end_time = time.time()
    print(f"Time taken to preallocate list of length {length}: {end_time - start_time} seconds")


"""
conclusion: numpy is much slower!
Time taken to preallocate numpy array of length 1000000: 0.2129201889038086 seconds
Time taken to preallocate list of length 1000000: 0.03465580940246582 seconds
"""

if __name__ == "__main__":
    main()