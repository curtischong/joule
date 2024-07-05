from dataset_service.datasets import AlexandriaDataset
from shared import root_dir

AlexandriaDataset().raw_data_to_lmdb(dataset_dir=f"{root_dir}/datasets/alexandria", output_dir=f"{root_dir}/datasets/lmdb/alexandria2")