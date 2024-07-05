from dataset_service.datasets import AlexandriaDataset
from shared import root_dir

# TODO: save to multiple lmdbs, and shard it so each file isn't too large
AlexandriaDataset().raw_data_to_lmdb(dataset_dir=f"{root_dir}/datasets/alexandria", output_dir=f"{root_dir}/datasets/lmdb/alexandria2")