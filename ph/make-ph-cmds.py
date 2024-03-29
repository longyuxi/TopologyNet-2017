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

"""Script for generating 'cmds.tmp' to be fed into GNU Parallel
I usually then run `time parallel -j 10 --eta < cmds.tmp` to run it
"""
# %%
from preprocessing import load_pdbbind_data_index
from pathlib import Path


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


# %% [markdown]
# ### Calculating the persistence diagram of each of the coordinates and storing them

# %%

index = load_pdbbind_data_index(index_location)

homology_calculator_location = script_base_path / 'calculate_homology.py'
pairwise_opposition_homology_calculator_location = script_base_path / 'calculate_pairwise_opposition_homologies_binned.py'
atom_2345_homology_calculator_location = script_base_path / 'calculate_2345_homology.py'

# Generates list of commands to be run by GNU parallel
# For each PDB entry, if the homology files for it don't exist, put a line into the command to calculate them

commands = ''
for idx, row in index.iterrows():
    pdb_name = row['PDB code']
    diagram_output_folder = Path(output_folder) / pdb_name
    diagram_output_folder.mkdir(parents=True, exist_ok=True)

    # input files
    pocket_pdb = base_folder / pdb_name / f'{pdb_name}_pocket.pdb'
    protein_pdb = base_folder / pdb_name / f'{pdb_name}_protein.pdb'
    ligand_mol2 = base_folder / pdb_name / f'{pdb_name}_ligand.mol2'

    # # protein
    # if not (diagram_output_folder / 'protein.pckl').is_file():
    #     commands += f'{python_interpreter} {homology_calculator_location} --input {str(protein_pdb)} --output {str(diagram_output_folder / "protein.pckl")}\n'

    # # pocket
    # if not (diagram_output_folder / 'pocket.pckl').is_file():
    #     commands += f'{python_interpreter} {homology_calculator_location} --input {str(pocket_pdb)} --output {str(diagram_output_folder / "pocket.pckl")}\n'

    # # ligand
    # if not (diagram_output_folder / 'ligand.pckl').is_file():
    #     commands += f'{python_interpreter} {homology_calculator_location} --input {str(ligand_mol2)} --output {str(diagram_output_folder / "ligand.pckl")}\n'

    # pairwise opposition homology
    if not (diagram_output_folder / 'pairwise_opposition.pckl').is_file():
        commands += f'{python_interpreter} {pairwise_opposition_homology_calculator_location} --protein {str(protein_pdb)} --ligand {str(ligand_mol2)} --output {str(diagram_output_folder / "pairwise_opposition.pckl")}\n'

    # 2345 homology
    if not (diagram_output_folder / '2345_homology.pckl').is_file():
        commands += f'{python_interpreter} {atom_2345_homology_calculator_location} --protein {str(protein_pdb)} --ligand {str(ligand_mol2)} --output {str(diagram_output_folder / "2345_homology.pckl")}\n'

with open('cmds.tmp', 'w') as cf:
    cf.write(commands)
