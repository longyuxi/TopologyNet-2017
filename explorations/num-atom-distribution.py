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

# %%
## For sampling perturbation
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'/ 'ph'))

# # Start from a PDB structure
# protein_pdb = '/home/longyuxi/Documents/mount/pdbbind-dataset/refined-set/3i3b/3i3b_protein.pdb'
# ligand_pdb = '/home/longyuxi/Documents/mount/pdbbind-dataset/refined-set/3i3b/3i3b_ligand.pdb'

from Bio.PDB import PDBParser, PDBIO
import calculate_pairwise_opposition_homologies_binned as h1
import calculate_2345_homology as h2345
from preprocessing import load_pdbbind_data_index
from tqdm import tqdm

io = PDBIO()
p = PDBParser(PERMISSIVE=1, QUIET=True)

##############################

# Location of index file
index_location = '/home/longyuxi/Documents/mount/pdbbind-dataset/index/INDEX_refined_data.2020'

# Where to output the computed homologies
output_folder = 'computed_homologies'

# Base folder of the refined set
base_folder = Path(index_location).parent.parent / 'refined-set'

# The interpreter with all the installed pre-requisite packages
python_interpreter = '/home/longyuxi/miniconda3/envs/donaldlab/bin/python'

# The folder where the scripts are located - Fine to leave as is
script_base_path = Path(__file__)

##############################
index = load_pdbbind_data_index(index_location)

def get_num_atoms(structure):
    i = 0
    for atom in structure.get_atoms():
        i += 1
    return i

num_atoms = []
for idx, row in tqdm(index.iterrows(), total=len(index)):
    pdb_name = row['PDB code']

    # input files
    pocket_pdb = base_folder / pdb_name / f'{pdb_name}_pocket.pdb'
    protein_pdb = base_folder / pdb_name / f'{pdb_name}_protein.pdb'
    ligand_mol2 = base_folder / pdb_name / f'{pdb_name}_ligand.mol2'

    protein_structure = p.get_structure(protein_pdb, protein_pdb)
    num_atoms.append(get_num_atoms(protein_structure))

# %%
import matplotlib.pyplot as plt
import numpy as np
plt.hist(num_atoms, bins=100)
plt.title(f'Distribution of number of atoms of structures in scPDB\nmean: {np.mean(num_atoms):.2f}, median: {np.median(num_atoms):.0f}, stdev: {np.std(num_atoms):.2f}')
plt.xlabel('# atoms')
plt.ylabel('# structures with this number of atoms')

