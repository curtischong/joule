train:
	python main.py --mode=train --config-yml=configs/s2ef/all/joule/upgraded_scn.yml --amp
create_mace_dataset_lmdb:
	python scripts/create_mace_dataset_lmdb.py