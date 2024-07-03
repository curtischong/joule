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
            # NOTE: curtis: we only update element_cnt by 1 if that element is in that sample
            # we are NOT updating by the actual count of atoms. This is probably better because
            # we're only interested in whether an atom type shows up in the sample or not (since it directly determines the size of our training dataset)
            unique_atomic_numbers = list(set(atomic_numbers))
            element_cnt.update(unique_atomic_numbers)

def main():
    element_cnt = Counter()
    for i in range(64):
        get_entries(element_cnt, IN_TRAIN_DIR, f"train_{i}")
    print(element_cnt)

if __name__ == "__main__":
    main()


"""
Most common elements by their counts (duplicate elements allowed if there's more than one element of it in the sample):
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

"""
Most common elements by their counts (duplicate elements in a sample are NOT allowed - they only contribute one to the count):
Counter({
    8: 865,235
    3: 221,922
    15: 173,303
    12: 165,092
    16: 152,386
    1: 141,750
    25: 138,426
    7: 133,405
    26: 130,435
    14: 130,341
    9: 126,709
    6: 110,454
    29: 102,564
    27: 99,441
    11: 97,806
    23: 95,044
    19: 92,705
    20: 90,141
    13: 85,810
    17: 84,359
    56: 80,777
    34: 75,325
    22: 73,879
    38: 70,742
    5: 70,635
    28: 64,889
    30: 63,410
    24: 62,255
    51: 60,008
    50: 57,016
    52: 55,687
    42: 53,093
    57: 52,526
    37: 52,157
    83: 51,038
    74: 48,765
    41: 48,415
    32: 47,970
    55: 46,078
    33: 44,548
    31: 43,849
    39: 43,469
    47: 41,305
    53: 38,171
    35: 37,565
    40: 34,429
    58: 34,216
    82: 33,560
    49: 32,924
    48: 32,136
    81: 30,562
    60: 30,500
    73: 29,736
    92: 28,311
    46: 26,465
    59: 26,071
    62: 25,530
    80: 23,103
    79: 22,580
    78: 22,528
    44: 21,767
    70: 21,517
    45: 21,097
    21: 20,543
    72: 20,420
    63: 20,159
    65: 19,273
    68: 19,157
    66: 19,149
    67: 18,344
    77: 17,635
    64: 15,626
    75: 15,603
    69: 13,740
    71: 13,045
    4: 11,608
    76: 10,334
    90: 10,262
    43: 5,800
    94: 4,455
    93: 3,765
    61: 2,549
    54: 2,058
    91: 1,619
    89: 1,367
    36: 178
    2: 62
    18: 6
})

"""

