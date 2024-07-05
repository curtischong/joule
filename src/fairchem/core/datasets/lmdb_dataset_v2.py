"""
Copyright (c) Meta, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import bisect
import logging
import pickle
import warnings
from pathlib import Path
from typing import TypeVar

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData

from fairchem.core.common.registry import registry
from fairchem.core.common.typing import assert_is_instance
from fairchem.core.common.utils import pyg2_data_transform
from fairchem.core.datasets._utils import rename_data_object_keys

T_co = TypeVar("T_co", covariant=True)


@registry.register_dataset("lmdb")
@registry.register_dataset("single_point_lmdb")
@registry.register_dataset("trajectory_lmdb")
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

        self.dbs = []
        self.key_prefix_sum_arr = [0]
        for path in self.db_paths:
            db = self.connect_db(self.path)
            num_entries = assert_is_instance(db.stat()["entries"], int)
            self.key_prefix_sum_arr.append(self.key_prefix_sum_arr[-1] + num_entries)
            self.dbs.append(db)


    def __len__(self) -> int:
        return self.key_prefix_sum_arr[-1]

    def __getitem__(self, idx: int) -> T_co:
        # if sharding, remap idx to appropriate idx of the sharded set
        if not self.path.is_file():
            # Figure out which db this should be indexed from.
            db_idx = bisect.bisect(self._keylen_cumulative, idx)
            # Extract index of element within that db.
            el_idx = idx
            if db_idx != 0:
                el_idx = idx - self._keylen_cumulative[db_idx - 1]
            assert el_idx >= 0

            # Return features.
            datapoint_pickled = (
                self.envs[db_idx]
                .begin()
                .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
            )
            data_object = pyg2_data_transform(pickle.loads(datapoint_pickled))
            data_object.id = f"{db_idx}_{el_idx}"
        else:
            datapoint_pickled = self.env.begin().get(
                f"{self._keys[idx]}".encode("ascii")
            )
            data_object = pyg2_data_transform(pickle.loads(datapoint_pickled))

        if self.key_mapping is not None:
            data_object = rename_data_object_keys(data_object, self.key_mapping)

        data_object.dataset_path = str(self.path)
        data_object.data_idx = idx

        return self.transforms(data_object)

    def connect_db(self, lmdb_path: Path | None = None) -> lmdb.Environment:
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
