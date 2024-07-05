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
from tqdm import tqdm
from dataset_service.datasets import AlexandriaDataset
from dataset_service.tools import int_to_bytes

from fairchem.core.common.registry import registry
from fairchem.core.common.typing import assert_is_instance
from fairchem.core.common.utils import pyg2_data_transform
from fairchem.core.datasets._utils import rename_data_object_keys
from fairchem.core.datasets.dataset_handler import DatasetHandler

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

        self.db_handlers: list[DatasetHandler] = []
        self.db_size_psa = [0] # psa = prefix sum array
        for path in self.db_paths:
            db_handler = DatasetHandler(path)
            self.db_handlers.append(db_handler)
            self.db_size_psa.append(self.db_size_psa[-1] + db_handler.num_entries)

        self.dataset_defs = {
            "alexandria": AlexandriaDataset(),
        }


    def __len__(self) -> int:
        return self.db_size_psa[-1]

    def __getitem__(self, idx: int) -> T_co:
        # Figure out which db this should be indexed from.
        db_idx = bisect.bisect(self.db_size_psa, idx)

        idx_in_db = idx
        if db_idx != 0:
            idx_in_db = idx - self.db_size_psa[db_idx - 1]
        assert idx_in_db >= 0

        data_object = self.db_handlers[db_idx].read_entry(idx_in_db)
        data_object.dataset_path = str(self.path)
        data_object.data_idx = idx
        return data_object
