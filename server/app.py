from ase import Atoms
from flask import Flask, request, jsonify
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from flask_cors import CORS
import json
from pymatgen.core.periodic_table import Element

app = Flask(__name__)
CORS(app)  # Allow all origins

checkpoint_path = "models/checkpoint_small.pt"
calculator = OCPCalculator(checkpoint_path=checkpoint_path)

most_common_elements_only_one_per_sample = [8, 3, 15, 12, 16, 1, 25, 7, 26, 14, 9, 6, 29, 27, 11, 23, 19, 20, 13, 17] 

def error_response(error_msg):
    response = jsonify(errorMsg=error_msg)
    response.status_code = 400
    return response


@app.route("/predict", methods=["POST"])
def calculate():
    data = request.get_json()
    crystal = data["crystal"]
    atomic_numbers = crystal["atomicNumbers"]

    # validate that we can handle each atomic number
    for atomic_number in atomic_numbers:
        if atomic_number > 118 or atomic_number < 1:
            return error_response(f"atomic number {atomic_number} is not a valid element")
        if atomic_number not in most_common_elements_only_one_per_sample:
            return error_response(f"the model is not trained to handle element {Element.from_Z(atomic_number)}")

    atoms = Atoms(symbols=atomic_numbers, scaled_positions=crystal["fracCoords"], cell=crystal["latticeMatrix"], pbc=[True, True, True])
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