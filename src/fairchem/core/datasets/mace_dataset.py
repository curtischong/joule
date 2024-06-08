from dataclasses import dataclass
import multiprocessing
from typing import List
from torch.utils.data import Dataset
import h5py
import numpy as np
import torch
from torch_geometric.data import Data

from diffusion.d3pm import D3PM
from diffusion.diffusion_helpers import VE_pbc

from diffusion.atomic_number_table import (
    atomic_numbers_to_indices,
    get_atomic_number_table_from_zs,
)
from fairchem.core.common.registry import registry


@registry.register_dataset("mace")
class AlexandriaDataset(Dataset):
    def __init__(self, dataset_config):
        super().__init__()
        # self.unique_atomic_numbers = set()
        dataset_path = dataset_config["src"]

        # self.configs = load_dataset(dataset_path)
        # for config in self.configs:
        #     self.unique_atomic_numbers.update(set(np.asarray(config.atomic_numbers)))

        # self.z_table = get_atomic_number_table_from_zs(
        #     [
        #         self.unique_atomic_numbers,
        #     ]
        # )
        # print(f"There are {len(self.z_table)} unique atomic numbers")

        # print(
        #     f"finished loading datasets {str(dataset_path)}. Found {len(self.configs)} entries"
        # )
        # # self.num_timesteps = model_attributes["num_timesteps"]

        dataset_path = "/home/ubuntu/joule/datasets/mptrj-gga-ggapu/mptrj-gga-ggapu"
        result = subprocess.run(f"ls -S {dataset_path}| head -n 1", shell=True, capture_output=True, text=True)

        # Print the result
        largest_filename = result.stdout.strip()
        def get_filename_number(filename):
            return filename[len("mp-"):][:-len(".extxyz")]

    # the first file is NOT 0. it starts at 1
    def 

    def __len__(self):
        return len(self.configs)

    def __getitem__(self, idx: int):
        # default_dtype = torch.float64  # for some reason, the default dtype is float32 in this subprocess. so I set it explicitly
        default_dtype = torch.float32  # Doing this because equiformer uses float32 linear layers. I don't know why. But if I have precision issues, I'll probably change this. The only reason why I'm okay with float32 is because we're not doing molecular dynamics
        config = self.configs[idx]

        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"NOTE: all of the data is only on gpu0")
        # device = torch.device(f"cuda:{0}")
        device = torch.device("cpu")

        # Get the initial state
        x_frac_start = torch.tensor(config.X0, dtype=default_dtype, device=device)
        atom_type_start = atomic_numbers_to_indices(
            self.z_table, config.atomic_numbers
        ).to(device)
        cell_start = torch.tensor(config.L0, dtype=default_dtype, device=device).view(
            -1, 3, 3
        )

        # # get the noisy state
        # timestep = torch.randint(1, self.num_timesteps + 1, size=(1,), device=device)

        # atom_type_noisy = self.atomic_diffusion.get_xt(atom_type_start, timestep)
        # x_frac_noisy, _wrapped_frac_eps_x, _used_sigmas = self.x_frac_diffusion.forward(
        #     x_frac_start,
        #     timestep.unsqueeze(0),
        #     cell_start.unsqueeze(0),
        #     torch.tensor(len(x_frac_start), dtype=torch.int, device=device).unsqueeze(
        #         0
        #     ),
        # )

        # x_cart_noisy = (x_frac_start @ cell_start).clone().detach().to(device)
        # # NOTE: we cannot fix the features at this point because when we add noise, the features will change

        res = Data(
            # x_cart_noisy=x_cart_noisy,
            x_frac_start=x_frac_start,
            # x_frac_noisy=x_frac_noisy,
            atom_type_start=atom_type_start,
            # atom_type_noisy=atom_type_noisy,
            cell_start=cell_start,
            natoms=torch.tensor(len(config.atomic_numbers), device=device),
            ith_sample=torch.tensor(idx, dtype=torch.float32),
            # timestep=timestep,
        )
        return res
