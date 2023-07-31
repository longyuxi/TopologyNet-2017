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