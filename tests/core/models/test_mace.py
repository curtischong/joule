from __future__ import annotations
import pytest
import torch
from ase.io import read
import os
from fairchem.core.preprocessing import AtomsToGraphs
from fairchem.core.common.utils import load_state_dict, setup_imports
import requests
import io
from fairchem.core.common.registry import registry
from fairchem.core.datasets import data_list_collater
import random
from fairchem.core.common.transforms import RandomRotate
import numpy as np
import logging

@pytest.fixture(scope="class")
def load_data(request):
    atoms = read(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "atoms.json"),
        index=0,
        format="json",
    )
    a2g = AtomsToGraphs(
        max_neigh=200,
        radius=6,
        r_edges=False,
        r_fixed=True,
    )
    data_list = a2g.convert_all([atoms])
    request.cls.data = data_list[0]


@pytest.fixture(scope="class")
def load_model(request):
    torch.manual_seed(4)
    setup_imports()

    model = registry.get_model_class("mace")(
        use_pbc=True,
        regress_forces=True,
        otf_graph=True,
        max_neighbors=20,
        # max_radius=12.0,
        max_num_elements=90,

        num_layers=8,
        lmax_list=[4],
        mmax_list=[2],
        sphere_channels=128,
        hidden_channels=128,
        num_sphere_samples=128,
        edge_channels=128,
        distance_function="gaussian",
        basis_width_scalar=2.0,
        distance_resolution=0.02,
        show_timing_info=True,
    )

    # Precision errors between mac vs. linux compound with multiple layers,
    # so we explicitly set the number of layers to 1 (instead of all 8).
    # The other alternative is to have different snapshots for mac vs. linux.
    model.num_layers = 1
    request.cls.model = model # here we are passing model in as a class-level fixture

class TestMace:
    @pytest.mark.usefixtures("load_data")
    @pytest.mark.usefixtures("load_model")
    def test_not_mixing_batch_dim(self):
        data = self.data.detach().clone()
        data2 = self.data.detach().clone() # clone so it's explicit that these are two diff tensors
        batch = data_list_collater([data, data2])

        # assert the model's weights init as 0 grad
        assert data.pos.grad == None, "data initialized with grad"
        assert data2.pos.grad == None, "data2 initialized with grad"
        batch.pos.requires_grad_(True)
        out = self.model(batch)

        loss = out["energy"][0] # the loss is only dependent on the first item in the batch
        loss.backward()

        num_atoms = batch.natoms[0]
        assert torch.all(batch.pos.grad[:num_atoms].ne(0)), "Expected all of the gradients to NOT be zero (in the first batch)" # note: there is a rare scenario where the grad IS 0. but it's so rare. do not care
        assert torch.all(batch.pos.grad[num_atoms:].eq(0)), "Expected all of the gradients to be zero (in the second batch)"


    # I think it's fine if we just test with on an untrained model
    @pytest.mark.usefixtures("load_data")
    @pytest.mark.usefixtures("load_model")
    def test_equivariance(self):
        random.seed(1)
        # Recreate the Data object to only keep the necessary features.
        data = self.data

        # Sampling a random rotation within [-180, 180] for all axes.
        transform = RandomRotate([-180, 180], [0, 1, 2])
        data_rotated, rot, inv_rot = transform(data.clone())
        assert not np.array_equal(data.pos, data_rotated.pos)

        # Pass it through the model.
        batch = data_list_collater([data, data_rotated])
        out = self.model(batch)

        # Compare predicted energies and forces (after inv-rotation).
        energies = out["energy"].detach().numpy() # convert to numpy, so more deciomals points are printed during a mismatch
        np.testing.assert_almost_equal(energies[0], energies[1], decimal=5)

        forces = out["forces"].detach()
        logging.info(forces)
        np.testing.assert_array_almost_equal(
            forces[: forces.shape[0] // 2].numpy(),
            torch.matmul(forces[forces.shape[0] // 2 :], inv_rot).numpy(),
            decimal=5,
        )