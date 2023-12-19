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

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

import glob
from models import WeiTopoNet
from dataset import process_homologies_into_tensor
from train import TRANSPOSE_DATASET
import torch

from ph.calculate_2345_homology import run as calculate_2345_homology
from ph.calculate_pairwise_opposition_homologies_binned import run as calculate_pairwise_opposition_homologies_binned

weights = glob.glob('/usr/project/dlab/Users/jaden/tnet2017-new/ml/trial/lightning_logs/version_3/checkpoints/*')[0]

def predict(protein_file, ligand_file, weights=weights):
    homologies = calculate_2345_homology(protein_file, ligand_file)
    pairwise_opposition_homologies = calculate_pairwise_opposition_homologies_binned(protein_file, ligand_file)
    pairwise_opposition_homologies = torch.from_numpy(pairwise_opposition_homologies).float()
    out_tensor = process_homologies_into_tensor([pairwise_opposition_homologies, homologies])

    wtn = WeiTopoNet()
    wtn = wtn.load_from_checkpoint(weights, transpose=TRANSPOSE_DATASET, map_location=torch.device('cpu'))

    return wtn(out_tensor[None, :, :])[0][0]



if __name__ == '__main__':
    from pathlib import Path
    example_pdbbind_folder = Path(__file__).parent / '..' / 'example_pdbbind_folder' / '1a1e'
    example_protein_file = (example_pdbbind_folder / '1a1e_protein.pdb').resolve()
    example_ligand_file = (example_pdbbind_folder / '1a1e_ligand.mol2').resolve()
    predicted_affinity = predict(str(example_protein_file), str(example_ligand_file))
    print(f'Predicted affinity: {predicted_affinity}')