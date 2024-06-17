from flask import Flask
import argparse
from fairchem.core.common.flags import flags
from fairchem.core.common.utils import (
    build_config,
)
from fairchem.core._cli import Runner

app = Flask(__name__)

parser: argparse.ArgumentParser = flags.get_parser()
args, override_args = parser.parse_known_args()
config = build_config(args, override_args)
Runner()(config)

@app.route("/")
def hello_world():
    return config

if __name__ == '__main__':
    app.run(debug=True) # TODO(curtis): make false