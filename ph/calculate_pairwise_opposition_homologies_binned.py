## This file is part of PATH, which is part of OSPREY 3.0
##
## OSPREY Protein Redesign Software Version 3.0
## Copyright (C) 2001-2023 Bruce Donald Lab, Duke University
##
## OSPREY is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License version 2
## as published by the Free Software Foundation.
##
## You should have received a copy of the GNU General Public License
## along with OSPREY.  If not, see <http://www.gnu.org/licenses/>.
##
## OSPREY relies on grants for its development, and since visibility
## in the scientific literature is essential for our success, we
## ask that users of OSPREY cite our papers. See the CITING_OSPREY
## document in this distribution for more information.
##
## Contact Info:
##    Bruce Donald
##    Duke University
##    Department of Computer Science
##    Levine Science Research Center (LSRC)
##    Durham
##    NC 27708-0129
##    USA
##    e-mail: www.cs.duke.edu/brd/
##
## <signature of Bruce Donald>, Mar 1, 2023
## Bruce Donald, Professor of Computer Science

# Calculates the pairwise opposition homology and stores it as a 4D np array
# Usage: python calculate_pairwise_opposition_homologies_binned.py --protein protein_file.pdb --ligand ligand_file.mol2 --output pl_opposition.pckl

from pathlib import Path
import sys
sys.path.append(str((Path(__file__).parent / '..' ).resolve()))
from joblib import Memory
memory = Memory("__joblib_cache__", verbose=0)

import numpy as np
from preprocessing import get_mol2_coordinates_by_element, get_pdb_coordinates_by_element
from gtda.homology import VietorisRipsPersistence
import argparse
import pickle

def opposition_homology(protein_coords, ligand_coords):
    if protein_coords is None or ligand_coords is None:
        return None

    def opposition_distance_metric(vec1, vec2):
        if np.abs(vec1[-1] - vec2[-1]) > 2:
            return np.linalg.norm(vec1[:3] - vec2[:3])
        else:
            return np.Inf

    # Append each coordinate with 1 for protein and 2 for ligand
    protein_coords = np.concatenate((protein_coords, np.ones((len(protein_coords), 1))), axis=1)
    ligand_coords = np.concatenate((ligand_coords, 4 * np.ones((len(ligand_coords), 1))), axis=1)

    combined_coords = np.concatenate((protein_coords, ligand_coords), axis=0)
    # Track connected components, loops, and voids
    homology_dimensions = [0, 1, 2]

    # Collapse edges to speed up H2 persistence calculation!
    persistence = VietorisRipsPersistence(
        metric=opposition_distance_metric,
        homology_dimensions=homology_dimensions,
        collapse_edges=True,
        max_edge_length=200
    )

    diagrams_basic = persistence.fit_transform(combined_coords[None, :, :])

    return diagrams_basic

def bin_persistence_diagram(diagram, bins=np.arange(0, 50, 0.25), types_of_homologies=3):
    output = np.zeros((len(bins) + 1, types_of_homologies, 3), dtype=np.int32)

    if diagram is None:
        return output

    # Bin persistence diagram
    # Binned persistence diagram be in the shape of (len(bins) + 1, types_of_homologies, 3)
    #   where the first channel corresponds to each entry,
    #   second channel is corresponds to type of homology (Betti-0, Betti-1, ...)
    #   third channel corresponds to (birth, persist, death)

    diagram = diagram[0]
    # Now diagram should be in shape (n, 3) where each row is
    #   (birth, death, type_of_homology)

    assert len(np.unique(diagram[:, 2])) == types_of_homologies

    for entry in diagram:
        homology_type = int(entry[2])
        # mark the beginning and end bins
        begin_bin = np.digitize(entry[0], bins)
        output[begin_bin, homology_type, 0] += 1
        end_bin = np.digitize(entry[1], bins)
        output[end_bin, homology_type, 2] += 1

        # mark the middle bins if there are any
        if not begin_bin == end_bin:
            output[np.arange(begin_bin + 1, end_bin), homology_type, 1] += 1

    return output

@memory.cache
def run(pdb_file, mol2_file):
    binned_diagrams = []
    # Make pairwise opposition homologies

    protein_heavy_elements = ['C', 'N', 'O', 'S']
    ligand_heavy_elements = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I']

    for pe in protein_heavy_elements:
        for le in ligand_heavy_elements:
            # calculate homology (pe, le)
            # opposition_homologies.append(...)
            protein_coords = get_pdb_coordinates_by_element(pdb_file, pe)
            ligand_coords = get_mol2_coordinates_by_element(mol2_file, le)
            diagram = opposition_homology(protein_coords, ligand_coords)
            binned_diagrams.append(bin_persistence_diagram(diagram))

    binned_diagrams = np.array(binned_diagrams)

    return binned_diagrams

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate persistence homology for up to dimension 2 using opposition distance')
    parser.add_argument('--protein', type=str, help='protein file (pdb)')
    parser.add_argument('--ligand', type=str, help='ligand file (mol2)')
    parser.add_argument('--output', type=str, help='Pickle file to output to')

    args = parser.parse_args()
    protein_pdb = Path(args.protein)
    ligand_mol2 = Path(args.ligand)
    output_file = Path(args.output)

    binned_diagrams = run(protein_pdb, ligand_mol2)

    with open(output_file, 'wb') as of:
        pickle.dump(binned_diagrams, of)
