from ase import Atoms
from flask import Flask, request
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow all origins

checkpoint_path = "models/checkpoint_small.pt"
calculator = OCPCalculator(checkpoint_path=checkpoint_path)

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