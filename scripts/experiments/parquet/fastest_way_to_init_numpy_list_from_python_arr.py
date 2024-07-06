import time
import numpy as np
import random

def main():
    length = 40000000

    start_time = time.time()
    # arr = list(range(length)) # using this is slower!
    arr = [random.randint(1, 100) for i in range(length)]
    numpy_array = np.array(arr)
    end_time = time.time()

    print(f"Time taken to create numpy array from python list: {end_time - start_time} seconds")

    start_time = time.time()
    numpy_array = np.empty(length, dtype=np.int32)
    for i in range(length):
        numpy_array[i] = random.randint(1, 100)
    end_time = time.time()
    print(f"Time taken to create numpy array from preallocating numpy array: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()

"""
# when assigning i to the array:
conclusion: using numpy.array is faster than preallocating a numpy array
Time taken to create numpy array from python list: 3.5242979526519775 seconds
Time taken to create numpy array from preallocating numpy array: 3.933303117752075 seconds

#  when assigning randint to the array:
python scripts/experiments/parquet/fastest_way_to_init_numpy_list_from_python_arr.py
Time taken to create numpy array from python list: 9.531171083450317 seconds
Time taken to create numpy array from preallocating numpy array: 9.733918905258179 seconds
"""