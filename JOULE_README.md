# setup:

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

sudo apt install python3.11
sudo apt install python3.11-venv
python3.11 -m venv venv

pip install torch submitit numpy torch-geometric PyYAML matplotlib==3.8.3 numba lmdb h5py pymatgen timm ase e3nn wandb tensorboard kaleido jupyterlab crystal-toolkit pytest syrupy tables polars seaborn

# to install this package so we have access to them in jupyter notebooks
pip install -e .


# visit https://github.com/rusty1s/pytorch_scatter

python -c "import torch; print(torch.__version__)"
pip install torch-scatter torch-cluster torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

How to get jupyter lab working:
https://chatgpt.com/share/63b7fae4-6c48-4231-a028-b01c9585ceec


download datasets:

run download_drive.py (be sure to change the path in the script so you're downloading the right dataset)

These datasets were from here:
mace train:
https://drive.google.com/drive/folders/12RsjlEtSlBldhoQVJapfC9lIFZwpnVk6
mace validation:
https://drive.google.com/drive/folders/10atLHo2VJyTcb9JqL8SjHZIornPntUh-