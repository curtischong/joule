import subprocess
from torch.utils.data import Dataset
import torch
from torch_geometric.data import Data
from fairchem.core.common.registry import registry
import numpy as np
import ase


@registry.register_dataset("mace")
class MaceDataset(Dataset):
    def __init__(self, dataset_config):
        super().__init__()
        # self.unique_atomic_numbers = set()
        dataset_path = dataset_config["src"]

        largest_filename_number = self._get_largest_file_number(dataset_path)

        ranges = self._generate_ranges(largest_filename_number)

        dataset_type = dataset_config["type"]
        if dataset_type == "train":
            indexes = np.arange(ranges[0])
        elif dataset_type == "val": # calida
            indexes = np.arange(ranges[1])
        else:
            raise f"unknown dataset type {dataset_type}"
        
        self.file_id_map = np.random.shuffle(indexes)

    def _generate_ranges(self, n, split_frac=[0.7, 0.15, 0.15]):
        assert sum(split_frac) == 1, "The split fractions must sum to 1."

        ranges = []
        start = 1 # the first file is starts at 1 NOT 0
        
        for frac in split_frac:
            end = start + int(n * frac)
            ranges.append((start, end))
            start = end
        
        # Adjust the last range to ensure it covers any remaining items due to rounding
        if end < n:
            ranges[-1] = (ranges[-1][0], n)
        
        return ranges


    def _get_filename_number(filename):
        return filename[len("mp-"):][:-len(".extxyz")]

    def _get_largest_file_number(self, dataset_path: str):
        result = subprocess.run(f"ls -S {dataset_path}| head -n 1", shell=True, capture_output=True, text=True)

        # Print the result
        largest_filename = result.stdout.strip()
        return self._get_filename_number(largest_filename)

    def __len__(self):
        return len(self.file_id_map)

    def __getitem__(self, idx: int):
        file_id = self.file_id_map[idx]

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
