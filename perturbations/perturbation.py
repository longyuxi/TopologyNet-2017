# %%
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'/ 'ph'))

# Different samplers
def simple_sampler(original_position, epsilon, idx):
    # Translates the original position in all six directions
    if idx >= 6: raise Exception
    new_position = original_position
    new_position[int(idx / 2)] += (idx%2) * epsilon
    return new_position


######################
# Example pdb structures
protein_pdb = '/home/longyuxi/Documents/mount/pdbbind-dataset/refined-set/1adl/1adl_protein.pdb'
ligand_mol2 = '/home/longyuxi/Documents/mount/pdbbind-dataset/refined-set/1adl/1adl_ligand.mol2'

# Sampler
# Returns a position based on original_position, idx
PERTURBATION_SAMPLER = simple_sampler
PERTURBATIONS_PER_ATOM = 6
#####################


from Bio.PDB import PDBParser, PDBIO
import calculate_pairwise_opposition_homologies_binned as h1
import calculate_2345_homology as h2345


pdbio = PDBIO()
pdbparser = PDBParser(PERMISSIVE=1, QUIET=True)

protein_structure = pdbparser.get_structure(protein_pdb, protein_pdb)


# Create perturbed versions of it and put them all in a database


def get_num_atoms(structure):
    i = 0
    for atom in structure.get_atoms():
        i += 1
    return i

n_atoms = get_num_atoms(protein_structure)
print(f'Number of atoms in protein: {n_atoms}')


# Do some fancy manipulation of structures here

pdbio.set_structure(protein_structure)
pdbio.save("out.pdb")