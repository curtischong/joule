from dataset_service.datasets import AlexandriaDataset
from shared import root_dir

# TODO: save to multiple lmdbs, and shard it so each file isn't too large
AlexandriaDataset().raw_data_to_lmdb(raw_dataset_input_dir=f"{root_dir}/datasets/alexandria", lmdb_output_dir=f"{root_dir}/datasets/lmdb/alexandria2")