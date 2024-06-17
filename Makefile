CONFIG_YML=configs/s2ef/all/joule/upgraded_escn.yml
LMDB_DATASET=datasets/lmdb/real_mace3/
ALEXANDRIA_DATASET=datasets/lmdb/

train1:
	python main.py --mode=train --config-yml=$(CONFIG_YML) --dataset.train.src=$(ALEXANDRIA_DATASET)alexandria_1_train.lmdb --dataset.val.src=$(ALEXANDRIA_DATASET)alexandria_1_val.lmdb --amp

train10:
	python main.py --mode=train --config-yml=$(CONFIG_YML) --dataset.train.src=$(ALEXANDRIA_DATASET)alexandria_10_train.lmdb --dataset.val.src=$(ALEXANDRIA_DATASET)alexandria_10_val.lmdb --amp
train1000:
	python main.py --mode=train --config-yml=$(CONFIG_YML) --dataset.train.src=$(ALEXANDRIA_DATASET)alexandria_1000_train.lmdb --dataset.val.src=$(ALEXANDRIA_DATASET)alexandria_1000_val.lmdb --amp

# 34 is the largest atomic number of the set of 20 most common elements
trainall:
	export NUMEXPR_MAX_THREADS=24 && python main.py --mode=train --config-yml=$(CONFIG_YML) --dataset.train.src=$(LMDB_DATASET)train --dataset.val.src=$(LMDB_DATASET)val --model.max_num_elements=34 --amp

create_mace_dataset_lmdb:
	python scripts/dataset_prep/create_mace_dataset_lmdb.py --config-yml=$(CONFIG_YML)

create_real_mace_dataset_lmdb:
	python scripts/dataset_prep/create_real_mace_dataset_lmdb2.py --config-yml=$(CONFIG_YML)

create_real_mace_dataset_lmdb3:
	python scripts/dataset_prep/create_real_mace_dataset_lmdb3.py

create_alexandria_dataset_lmdb:
	python scripts/dataset_prep/create_alexandria_dataset_lmdb.py --config-yml=$(CONFIG_YML)

predict:
	python main.py --mode=predict --config-yml=$(CONFIG_YML) --dataset.test.src=$(LMDB_DATASET)train/1.lmdb --checkpoint=models/last.pt --amp --optim.eval_batch_size=1

# set the batch size so when predicting, we don't use as much memory
run_server:
	python server/app.py --host=0.0.0.0 --mode=predict --config-yml=$(CONFIG_YML) --checkpoint=models/last.pt --amp  --optim.eval_batch_size=1

run_devserver:
	python server/app.py --mode=predict-web --config-yml=$(CONFIG_YML) --checkpoint=models/last.pt --amp  --optim.eval_batch_size=1