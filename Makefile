train1:
	python main.py --mode=train --config-yml=configs/s2ef/all/joule/upgraded_escn.yml --dataset.train.src=datasets/lmdb/alexandria_1_train.lmdb --dataset.val.src=datasets/lmdb/alexandria_1_val.lmdb --amp
train10:
	python main.py --mode=train --config-yml=configs/s2ef/all/joule/upgraded_escn.yml --dataset.train.src=datasets/lmdb/alexandria_10_train.lmdb --dataset.val.src=datasets/lmdb/alexandria_10_val.lmdb --amp
train1000:
	python main.py --mode=train --config-yml=configs/s2ef/all/joule/upgraded_escn.yml --dataset.train.src=datasets/lmdb/alexandria_1000_train.lmdb --dataset.val.src=datasets/lmdb/alexandria_1000_val.lmdb --amp
create_mace_dataset_lmdb:
	python scripts/dataset_prep/create_mace_dataset_lmdb.py --config-yml=configs/s2ef/all/joule/upgraded_escn.yml
create_alexandria_dataset_lmdb:
	python scripts/dataset_prep/create_alexandria_dataset_lmdb.py --config-yml=configs/s2ef/all/joule/upgraded_escn.yml