import numpy as np
from ase.db import connect
from ase import data
import argparse


def load_atomrefs(datafile):
    tmp_atomref = {}

    with open(datafile, 'r') as dfile:
        for line in dfile:
            atom_type, energy = line.split()
            tmp_atomref[data.atomic_numbers[atom_type]] = float(energy)

    max_elem = max(tmp_atomref.keys())

    atomref = np.zeros((max_elem + 1, 1))

    for elem in tmp_atomref.keys():
        atomref[elem] = tmp_atomref[elem]

    return atomref.tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Add atomrefs to existing ASE database.")
    parser.add_argument("database", type=str, help="Path to ASE database.")
    parser.add_argument("atomref", type=str, help="File with atomrefs.")
    args = parser.parse_args()

    with connect(args.database) as db:

        # Get atomrefs
        atomref = load_atomrefs(args.atomref)

        # Get metadata
        metadata = db.metadata

        # Update
        metadata["atref_labels"] = ["energy"]#,"diagonal_energies"]
        metadata["atomrefs"] = atomref

        # Store
        db.metadata = metadata
