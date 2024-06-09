train:
	python main.py --mode=train --config-yml=configs/s2ef/all/joule/upgraded_escn.yml --amp
create_mace_dataset_lmdb:
	python scripts/create_mace_dataset_lmdb.py --config-yml=configs/s2ef/all/joule/upgraded_escn.yml