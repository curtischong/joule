import lmdb
import time

# The goal: to save as mucuh data in as most compressed a form possible
# we should be structuring the data

dataset_path = "test_db.lmdb"

def main():
    db = lmdb.open(
        dataset_path,
        map_size=1099511627776 * 2, # two terabytes is the max size of the db
        subdir=False,
        meminit=False,
        map_async=True,
    )

    start_time = time.time()

    print(f"creating {dataset_path}")
    num_samples = len(atoms)

    for fid, data in tqdm(enumerate(atoms), total=num_samples):
        positions = torch.Tensor(data["positions"])
        cell = torch.Tensor(data["cell"]).view(1, 3, 3)
        atomic_numbers = torch.Tensor(data["atomic_numbers"])
        natoms = positions.shape[0]

        data = Data(
            cell=cell,
            pos=positions,
            atomic_numbers=atomic_numbers,
            natoms=natoms,
            fid=torch.LongTensor([fid]),
            fixed=torch.full((natoms,), 0, dtype=torch.int8), # make all the atoms fixed, so the model's prediction for each atom contributes to the loss
            energy=torch.Tensor([data["energy"]]),
            forces=torch.Tensor(data["forces"]),
        )

        txn = db.begin(write=True)
        txn.put(f"{fid}".encode("ascii"), pickle.dumps(data, protocol=-1))
        txn.commit()

    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(num_samples, protocol=-1))
    txn.commit()


    db.sync()
    db.close()

    end_time = time.time()

if __name__ == "__main__":
    main()