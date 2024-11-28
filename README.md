# Joule

This is an equivariant neural network potential (using the Escn architecture) and is trained on a subset of the mace-mp-0 dataset.

Here are the parameters of the training run (8 hours on an A100-80GB GPU)
https://github.com/curtischong/joule/pull/45/files. I performed this run to estimate how long it would take to train a full model that would be competitive on Matbench. I stopped the training early since it was a test. Also, I only trained the model on a subset of the full dataset (20 most "common" atoms). Results here: https://wandb.ai/curtischong/joule/workspace?nw=nwusercurtischong

### Data prep
- I downloaded data from https://drive.google.com/drive/folders/12RsjlEtSlBldhoQVJapfC9lIFZwpnVk6 (Note you need your own Google drive API credentials. I put mine inside a .json file under this path: scripts/dataset_download/credentials.json)

- To train the model with the most amount of data but with only a small subset of atoms, I wrote a script to find the number of times each atom appeared in a single material (e.g. if hydrogen appeared 10 times in a material, then it would be counted as 1 - since that is one training example). view scripts/dataset_eda/mace/most_common_elements.py

After I found the 20 elements that would yield the most training samples for the model, I ran scripts/dataset_prep/v1_format/create_real_mace_dataset_lmdb3.py to prep the dataset. Note: there were many exact duplicate systems between the train and validation datasets that this script removes)


### Training

Run `make trainall` to train the model on a GPU. In general, I used a subset of the Alexandria dataset for local development and the mace-mp-0 dataset to train the real model on the A100 GPU.

<h1 align="center"> <code>fairchem</code> by FAIR Chemistry </h1>

<p align="center">
  <img width="559" height="200" src="https://github.com/FAIR-Chem/fairchem/assets/45150244/5872c21c-8f39-41af-b703-af9817f0affe"?
</p>


<h4 align="center">

![tests](https://github.com/FAIR-Chem/fairchem/actions/workflows/test.yml/badge.svg?branch=main)
[![documentation](https://github.com/FAIR-Chem/fairchem/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/FAIR-Chem/fairchem/actions/workflows/docs.yml)
[![Static Badge](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)

</h4>

`fairchem` is the [FAIR](https://ai.meta.com/research/) Chemistry's centralized repository of all its data, models, demos, and application efforts for materials science and quantum chemistry.

### Documentation
If you are looking for `Open-Catalyst-Project/ocp`, it can now be found at [`fairchem.core`](src/fairchem/core). Visit its corresponding documentation [here](https://fair-chem.github.io/).

### Contents
The repository is organized into several directories to help you find what you are looking for:

- [`fairchem.core`](src/fairchem/core): State of the art machine learning models for materials science and chemistry
- [`fairchem.data`](src/fairchem/data): Dataset downloads and input generation codes
- [`fairchem.demo`](src/fairchem/demo): Python API for the [Open Catalyst Demo](https://open-catalyst.metademolab.com/)
- [`fairchem.applications`](src/fairchem/applications): Follow up applications and works (AdsorbML, CatTSunami, etc.)

### Installation
Packages can be installed in your environment by the following:
```
pip install -e packages/fairchem-{fairchem-package-name}
```

`fairchem.core` requires you to first create your environment
- [Installation Guide](https://fair-chem.github.io/core/install.html)

CURTIS' NOTE: READ THE BOTTOM FOR THE REAL INSTALLATION INSTRUCTIONS

### Quick Start
Pretrained models can be used directly with ASE through our `OCPCalculator` interface:

```python
from ase.build import fcc100, add_adsorbate, molecule
from ase.optimize import LBFGS
from fairchem.core import OCPCalculator

# Set up your system as an ASE atoms object
slab = fcc100('Cu', (3, 3, 3), vacuum=8)
adsorbate = molecule("CO")
add_adsorbate(slab, adsorbate, 2.0, 'bridge')

calc = OCPCalculator(
    model_name="EquiformerV2-31M-S2EF-OC20-All+MD",
    local_cache="pretrained_models",
    cpu=False,
)
slab.calc = calc

# Set up LBFGS dynamics object
dyn = LBFGS(slab)
dyn.run(0.05, 100)
```

If you are interested in training your own models or fine-tuning on your datasets, visit the [documentation](https://fair-chem.github.io/) for more details and examples.

### Why a single repository?
Since many of our repositories rely heavily on our other repositories, a single repository makes it really easy to test and ensure consistency across repositories. This should also help simplify the installation process for users who are interested in integrating many of the efforts into one place.

### LICENSE
`fairchem` is available under a [MIT License](LICENSE.md).


### Real Installation
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

sudo apt install python3.11
sudo apt install python3.11-venv
python3.11 -m venv venv

pip install torch submitit ocpmodels numpy torch-geometric PyYAML matplotlib==3.8.3 numba lmdb h5py pymatgen timm ase e3nn wandb tensorboard kaleido jupyterlab crystal-toolkit brotli

# to install this package so we have access to them in jupyter notebooks
pip install -e .


# visit https://github.com/rusty1s/pytorch_scatter

python -c "import torch; print(torch.__version__)"
pip install torch-scatter torch-cluster torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```