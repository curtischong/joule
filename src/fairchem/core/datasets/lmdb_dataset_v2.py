"""
Copyright (c) Meta, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import TypeVar

from torch.utils.data import Dataset
from dataset_service.datasets import AlexandriaDataset

from fairchem.core.common.registry import registry
from fairchem.core.datasets.dataset_handler import LmdbDatasetHandler

T_co = TypeVar("T_co", covariant=True)


@registry.register_dataset("lmdb_v2")
class LmdbDatasetV2(Dataset[T_co]):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        path = Path(self.config["src"])

        if path.is_file():
            self.db_paths = [path]
        else:
            self.db_paths = sorted(self.path.glob("*.lmdb"))
            assert len(self.db_paths) > 0, f"No LMDBs found in '{self.path}'"

        self.db_handlers: list[LmdbDatasetHandler] = []
        self.db_size_psa = [0] # psa = prefix sum array
        for path in self.db_paths:
            db_handler = LmdbDatasetHandler(path)
            self.db_handlers.append(db_handler)
            self.db_size_psa.append(self.db_size_psa[-1] + db_handler.num_entries)

    def __len__(self) -> int:
        return self.db_size_psa[-1]

    def __getitem__(self, idx: int) -> T_co:
        # Figure out which db this should be indexed from.
        db_idx = bisect.bisect(self.db_size_psa, idx)

        idx_in_db = idx
        if db_idx != 0:
            idx_in_db = idx - self.db_size_psa[db_idx - 1]
        assert idx_in_db >= 0

        # TODO(curtis): handle tags and everything else
        data_object = self.db_handlers[db_idx].read_entry(idx_in_db)
        data_object.dataset_path = str(self.path)
        data_object.data_idx = idx
        return data_object

    def close_db(self) -> None:
        for handler in self.db_handlers:
            handler.close_db()