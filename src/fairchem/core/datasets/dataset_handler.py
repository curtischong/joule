from dataset_service.datasets import AlexandriaDataset
import lmdb
from pathlib import Path

from fairchem.core.common.typing import assert_is_instance

# used when reading lmdb datasets. it tracks dataset info so you can easily read the data
class LmdbDatasetHandler:
    def __init__(self, dataset_path: str):
        if dataset_path.contains("alexandria"):
            self.dataset_def = AlexandriaDataset()
        else:
            raise NotImplementedError(f"Dataset {dataset_path} not supported")

        self.db = self._connect_db(dataset_path)
        self.num_entries = assert_is_instance(self.db.stat()["entries"], int)

    def read_entry(self, idx: int):
        return self.dataset_def.read_entry(self.db, idx)

    def _connect_db(self, lmdb_path: Path | None = None) -> lmdb.Environment:
        return lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=1,
        )

    def close_db(self) -> None:
        if not self.path.is_file():
            for env in self.envs:
                env.close()
        else:
            self.env.close()