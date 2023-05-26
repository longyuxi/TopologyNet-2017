# Initial prototype: Calculate just the atomic persistence diagram
# Later: Move on to use pairiwise opposition homology as detailed in Wei's paper

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'/ 'ph'))

import numpy as np

from preprocessing import load_pdbbind_data_index, get_mol2_coordinates_by_element, get_pdb_coordinates_by_element
from calculate_pairwise_opposition_homologies_binned import opposition_homology
from calculate_2345_homology import atom_persistence_homology

################

platform = 'DCC'

if platform == 'local':
    INDEX_LOCATION = Path('/home/longyuxi/Documents/mount/pdbbind-dataset/index/INDEX_refined_data.2020')
    BASE_FOLDER = Path('/home/longyuxi/Documents/mount/pdbbind-dataset/refined-set')
    OUTPUT_BASE_FOLDER = Path('/home/longyuxi/Documents/mount/persistence-diagrams')
elif platform == 'DCC':
    INDEX_LOCATION = Path('/hpc/group/donald/yl708/pdbbind/index/INDEX_refined_data.2020')
    BASE_FOLDER = Path('/hpc/group/donald/yl708/pdbbind/refined-set')
    OUTPUT_BASE_FOLDER = Path('/hpc/group/donald/yl708/persistence-diagrams')

################

INDEX = load_pdbbind_data_index(INDEX_LOCATION)

def get_persistence_diagrams(pdb_file, mol2_file):
    protein_heavy_elements = ['C', 'N', 'O', 'S']
    ligand_heavy_elements = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I']

    # For testing purposes
    # protein_heavy_elements = ['C']
    # ligand_heavy_elements = ['C']

    diagrams = []

    for pe in protein_heavy_elements:
        for le in ligand_heavy_elements:
            # calculate homology (pe, le)
            # opposition_homologies.append(...)
            protein_coords = get_pdb_coordinates_by_element(pdb_file, pe)
            ligand_coords = get_mol2_coordinates_by_element(mol2_file, le)
            diagram = opposition_homology(protein_coords, ligand_coords)
            diagrams.append(diagram)

    return diagrams


def get_2345_persistence_diagrams(pdb_file, mol2_file):

    homologies = [] # this is used to store all of the calculated persistence diagrams

    def concatenate_coordinates(list_of_coordinates):
        # input: list of ndarray of size (*, 3)
        output = None
        for i in range(len(list_of_coordinates) - 1):
            if i == 0:
                output = np.concatenate((list_of_coordinates[i], list_of_coordinates[i+1]))
            else:
                output = np.concatenate((output, list_of_coordinates[i+1]))

        return output

    protein_heavy_elements = ['C', 'N', 'O', 'S']
    ligand_heavy_elements = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I']

    # 2: all heavy atoms of protein
    protein_heavy_atom_coords = []
    for pe in protein_heavy_elements:
        protein_coords = get_pdb_coordinates_by_element(pdb_file, pe)
        protein_heavy_atom_coords.append(protein_coords)

    protein_heavy_atom_coords = concatenate_coordinates(protein_heavy_atom_coords)

    homologies.append(atom_persistence_homology(protein_heavy_atom_coords))

    # 3: all heavy atoms of protein and all heavy atoms of ligand
    ligand_heavy_atom_coords = []
    for le in ligand_heavy_elements:
        ligand_coords = get_mol2_coordinates_by_element(mol2_file, le)
        ligand_heavy_atom_coords.append(ligand_coords)

    ligand_heavy_atom_coords = concatenate_coordinates(ligand_heavy_atom_coords)
    all_heavy_atom_coords = np.concatenate((protein_heavy_atom_coords, ligand_heavy_atom_coords))

    homologies.append(atom_persistence_homology(all_heavy_atom_coords))

    # 4: all carbon atoms of protein
    protein_carbon_coords = get_pdb_coordinates_by_element(pdb_file, 'C')
    homologies.append(atom_persistence_homology(protein_carbon_coords))

    # 5: all carbon atoms of protein and all heavy atoms of ligand
    ligand_carbon_coords = get_mol2_coordinates_by_element(mol2_file, 'C')
    all_carbon_coords = np.concatenate((protein_carbon_coords, ligand_carbon_coords))
    homologies.append(atom_persistence_homology(all_carbon_coords))

    return homologies

def main():

    train_set = []

    for idx, row in INDEX.iterrows():
        pdb_name = row['PDB code']
        diagram_output_folder = OUTPUT_BASE_FOLDER / pdb_name
        diagram_output_folder.mkdir(parents=True, exist_ok=True)

        # input files
        pocket_pdb = BASE_FOLDER / pdb_name / f'{pdb_name}_pocket.pdb'
        protein_pdb = BASE_FOLDER / pdb_name / f'{pdb_name}_protein.pdb'
        ligand_mol2 = BASE_FOLDER / pdb_name / f'{pdb_name}_ligand.mol2'

        diagrams = get_2345_persistence_diagrams(protein_pdb, ligand_mol2)

        print(diagrams)

        break






if __name__ == '__main__':
    main()