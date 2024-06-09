train1:
	python main.py --mode=train --config-yml=configs/s2ef/all/joule/upgraded_escn.yml --dataset.train.src=datasets/lmdb/mace_1_train.lmdb --dataset.val.src=datasets/lmdb/mace_1_val.lmdb --amp
train10:
	python main.py --mode=train --config-yml=configs/s2ef/all/joule/upgraded_escn.yml --dataset.train.src=datasets/lmdb/mace_10_train.lmdb --dataset.val.src=datasets/lmdb/mace_10_val.lmdb --amp
train1000:
	python main.py --mode=train --config-yml=configs/s2ef/all/joule/upgraded_escn.yml --dataset.train.src=datasets/lmdb/mace_1000_train.lmdb --dataset.val.src=datasets/lmdb/mace_1000_val.lmdb --amp
create_mace_dataset_lmdb:
	python scripts/create_mace_dataset_lmdb.py --config-yml=configs/s2ef/all/joule/upgraded_escn.yml