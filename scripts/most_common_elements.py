# the reason why we have this file is because we want to only train the model on the most common elements in the dataset
# so we maximize the number of trainin gsamples for as few elements (so the model is more accurate for a small number of elements)

import h5py
from tqdm import tqdm
from collections import Counter

IN_TRAIN_DIR = "datasets/real_mace/train"

def get_entries(element_cnt, in_dir, file_name):
    with h5py.File(f"{in_dir}/{file_name}.h5", 'r') as hdf5_file:
        num_configs = len(hdf5_file["config_batch_0"])
        for i in tqdm(range(num_configs)):
            config_group = hdf5_file[f'config_batch_0/config_{i}']
            atomic_numbers = config_group['atomic_numbers'][:]
            element_cnt.update(atomic_numbers)

def main():
    element_cnt = Counter()
    for i in range(64):
        get_entries(element_cnt, IN_TRAIN_DIR, f"train_{i}")
    print(element_cnt)

if __name__ == "__main__":
    main()


"""
Counter({
    8: 18230139,
    1: 3022596,
    9: 1852893,
    16: 1671077,
    15: 1286606,
    7: 1270037,
    3: 1256572,
    12: 1236003,
    6: 1062038,
    14: 1044832,
    17: 941778,
    26: 747534,
    34: 735018,
    5: 673665,
    25: 666595,
    13: 607139,
    11: 565106,
    23: 483833,
    19: 465497,
    27: 462438,
    29: 443035,
    20: 432550,
    22: 379561,
    35: 378763,
    53: 378382,
    56: 377169,
    30: 369533,
    52: 356216,
    28: 349020,
    38: 339944,
    51: 325179,
    42: 302771,
    83: 289679,
    32: 280517,
    50: 280144,
    33: 274270,
    31: 266735,
    24: 261169,
    57: 249561,
    41: 242811,
    37: 234255,
    47: 205230,
    74: 199139,
    55: 187873,
    82: 178170,
    40: 170115,
    39: 169719,
    48: 159085,
    49: 153770,
    73: 152438,
    60: 144374,
    58: 138820,
    81: 131571,
    80: 131387,
    59: 124528,
    46: 119110,
    92: 110539,
    62: 109541,
    70: 103595,
    45: 102423,
    78: 93153,
    65: 92260,
    21: 92051,
    68: 91587,
    44: 90797,
    79: 88249,
    66: 85794,
    67: 84252,
    75: 79724,
    72: 77566,
    4: 71453,
    77: 69712,
    63: 65941,
    69: 58929,
    64: 57965,
    71: 56551,
    76: 42923,
    90: 35872,
    43: 16441,
    94: 15806,
    54: 12639,
    93: 11484,
    61: 6095,
    89: 3206,
    91: 3002,
    36: 1103,
    2: 376,
    18: 12
})
"""