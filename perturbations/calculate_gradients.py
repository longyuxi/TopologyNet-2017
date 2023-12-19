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

## Calculates the change in prediction with respect to each perturbation

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'/ 'ml'))
sys.path.append(str(Path(__file__).parent / '..'/ 'ph'))
from dispatch_jobs import DB
import torch


######################################
KEY_PREFIX = 'perturb_1a1e_'
WEIGHTS = '/hpc/group/donald/yl708/TopologyNet-2017/ml/lightning_logs/version_0/checkpoints/epoch=135-step=26656.ckpt'

######################################

def get_data(base_path, transpose=False) -> torch.Tensor:
    # An adaptation of ml/dataset.py
    with open(Path(base_path) / 'pairwise_opposition.pckl', 'rb') as f:
        pw_opposition= torch.tensor(pickle.load(f), dtype=torch.float)
        pw_opposition = pw_opposition[:, :, 0, 0] # now should be of shape torch.Size([36, 201])    
    
    with open(Path(base_path) / '2345_homology.pckl', 'rb') as f:
        raw_2345_homologies: np.ndarray = pickle.load(f)

        final_2345_homologies = [] # used as output
        for hom in raw_2345_homologies:
            # iterate through 2345 and doing the same thing

            # hom is of shape ndarray(201, 3, 3)
            hom = hom[:, 1:3, :] # shape ndarray(201, 2, 3)
            hom = hom.reshape(201, 6)

            final_2345_homologies.append(hom)

        # A lot of reshaping to get it into shape (24, 201)
        final_2345_homologies = np.array(final_2345_homologies) # (4, 201, 6)
        pl_heavy_diff = final_2345_homologies[0] - final_2345_homologies[1]
        pl_carbon_diff = final_2345_homologies[2] - final_2345_homologies[3]
        final_2345_homologies = np.concatenate((final_2345_homologies, pl_heavy_diff[None, :, :], pl_carbon_diff[None, :, :]), 0) # (6, 201, 6)
        final_2345_homologies = np.swapaxes(final_2345_homologies, 0, 1) # (201, 6, 6)
        final_2345_homologies = final_2345_homologies.reshape(201, 36)
        final_2345_homologies = np.swapaxes(final_2345_homologies, 0, 1)

        final_2345_homologies = torch.tensor(final_2345_homologies, dtype=torch.float)
        
    out = torch.concat((pw_opposition, final_2345_homologies), dim=0)

    if transpose:
        out = out.T

    return out

if __name__ == '__main__':
    import numpy as np
    from tqdm import tqdm
    import pickle

    from models import WeiTopoNet

    print(f'Calculating gradients using key prefix: {KEY_PREFIX}')
    database_keys = DB.keys(KEY_PREFIX + '*')

    print(f'Loading weights from {WEIGHTS}')
    wtn = WeiTopoNet()
    wtn = wtn.load_from_checkpoint(WEIGHTS, transpose=True)

    # For each entry, run
    for key in tqdm(database_keys):
        info = DB.hgetall(key)

        if info['finished'] == 'True' and info['error'] == 'False':
            base_path = info['save_folder']

            data = get_data(base_path, transpose=True)
            data = data.cuda(0)

            # input size: [201, 72]
            y_hat = wtn(data[None, :, :])[0][0].detach().cpu().numpy().item()

            # Put y_hat in the database
            info['y_hat'] = y_hat

            DB.hset(key, mapping=info)

