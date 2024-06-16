from fairchem.core.common.utils import load_config
import argparse

def parse_config():
    parser = argparse.ArgumentParser(
        description="Graph Networks for Electrocatalyst Design"
    )
    parser.add_argument(
        "--config-yml", type=str, help="pass in the config so we know the cutoff radius and max neighbors for the graph"
    )
    args, override_args = parser.parse_known_args()
    config, duplicates_warning, duplicates_error = load_config(args.config_yml)
    if len(duplicates_warning) > 0:
        print(f"Warning: Duplicate keys found in config file: {duplicates_warning}")
    assert len(duplicates_error) == 0, "Errors found in config file"
    return config

def get_range(n: int, dataset_type: str):
    # shuffle the system paths so when we generate the ranges, we ahve a good mix of all the datapoints
    if dataset_type == "all":
        return generate_ranges(n, split_frac=[0.7, 0.15, 0.15], start_at_1=True)
    elif dataset_type == "10000":
        assert n >= 30000, "The dataset is too small to generate ranges for 10000 datapoints"
        return [[0, 10000], [10000, 20000], [20000, 30000]]
    elif dataset_type == "1000":
        assert n >= 3000, "The dataset is too small to generate ranges for 1000 datapoints"
        return [[0, 1000], [1000, 2000], [2000, 3000]]
    elif dataset_type == "100":
        assert n >= 300, "The dataset is too small to generate ranges for 100 datapoints"
        return [[0, 100], [100, 200], [200, 300]]
    elif dataset_type == "10":
        assert n >= 30, "The dataset is too small to generate ranges for 10 datapoints"
        return [[0, 10], [10, 20], [20, 30]]
    elif dataset_type == "1":
        assert n >= 3, "The dataset is too small to generate ranges for 1 datapoint"
        return [[0, 1], [1, 2], [2, 3]]
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def generate_ranges(n:int, split_frac=[0.7, 0.15, 0.15], start_at_1=True):
    assert sum(split_frac) == 1, "The split fractions must sum to 1."

    ranges = []
    if start_at_1:
        start = 1 # have the option to start at 1 since the first file in the mace-mp-0 dataset starts at 1 NOT 0
    else:
        start = 0
    
    for frac in split_frac:
        end = start + int(n * frac)
        ranges.append((start, end))
        start = end
    
    # Adjust the last range to ensure it covers any remaining items due to rounding
    if end < n:
        ranges[-1] = (ranges[-1][0], n)
    
    return ranges