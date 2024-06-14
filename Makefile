CONFIG_YML=configs/s2ef/all/joule/upgraded_escn.yml
LMDB_DATASET=datasets/lmdb/real_mace/

train1:
	python main.py --mode=train --config-yml=$(CONFIG_YML) --dataset.train.src=$(LMDB_DATASET)1_train.lmdb --dataset.val.src=$(LMDB_DATASET)1_val.lmdb --amp
train10:
	python main.py --mode=train --config-yml=$(CONFIG_YML) --dataset.train.src=$(LMDB_DATASET)10_train.lmdb --dataset.val.src=$(LMDB_DATASET)10_val.lmdb --amp
train1000:
	python main.py --mode=train --config-yml=$(CONFIG_YML) --dataset.train.src=$(LMDB_DATASET)1000_train.lmdb --dataset.val.src=$(LMDB_DATASET)1000_val.lmdb --amp
create_mace_dataset_lmdb:
	python scripts/dataset_prep/create_mace_dataset_lmdb.py --config-yml=$(CONFIG_YML)
create_real_mace_dataset_lmdb:
	python scripts/dataset_prep/create_real_mace_dataset_lmdb2.py --config-yml=$(CONFIG_YML)
create_alexandria_dataset_lmdb:
	python scripts/dataset_prep/create_alexandria_dataset_lmdb.py --config-yml=$(CONFIG_YML)
