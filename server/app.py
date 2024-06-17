from ase import Atoms
from flask import Flask, request
# import argparse
# from fairchem.core.common.flags import flags
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
# from fairchem.core.common.utils import (
#     build_config,
# )
# from fairchem.core._cli import Runner
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow all origins

# parser: argparse.ArgumentParser = flags.get_parser()
# args, override_args = parser.parse_known_args()
# config = build_config(args, override_args)
# Runner()(config)
config_yml = "configs/s2ef/all/joule/upgraded_escn.yml"
checkpoint_path = "models/checkpoint_small.pt"
calculator = OCPCalculator(
    # config_yml=config_yml,
    checkpoint_path=checkpoint_path,
    # trainer="forces"
)

@app.route("/predict", methods=["POST"])
def calculate():
    data = request.get_json()
    crystal = data["crystal"]
    atoms = Atoms(symbols=crystal["atomicNumbers"], scaled_positions=crystal["fracCoords"], cell=crystal["latticeMatrix"], pbc=[True, True, True])
    calculator.calculate(atoms, properties=["energy", "forces"], system_changes=[]) # I don't even think that system_changes is used
    energy = calculator.results["energy"]
    forces = calculator.results["forces"].tolist()
    # print(energy, forces)
    return {"energy": energy, "forces": forces}

@app.route("/")
def hello_world():
    return "hi"

if __name__ == '__main__':
    app.run(debug=True) # TODO(curtis): make false