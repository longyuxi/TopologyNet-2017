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
sys.path.append(str(Path(__file__).parent / '..'/ 'ph'))

import numpy as np
import pytorch_lightning as pl
from preprocessing import load_pdbbind_data_index
import plotly.express as px

from models import WeiTopoNet
from dataset import ProteinHomologyDataModule

###################
# Change these

index_location = '/home/longyuxi/Documents/mount/pdbbind-dataset/index/INDEX_refined_data.2020'
homologies_base_folder = (Path(__file__).parent / '..' / 'ph'/ 'computed_homologies').resolve()
homologies_base_folder = str(homologies_base_folder)

# See documentation of `WeiTopoNet` in models.py for details on this
TRANSPOSE_DATASET = False

##################

pl.seed_everything(123)

print('transpose dataset:', TRANSPOSE_DATASET)

def get_train_test_indices():
    index = load_pdbbind_data_index(index_location)

    # Randomly train-test split
    index_shuffled = index.iloc[np.random.permutation(len(index))]
    index_shuffled.reset_index(drop=True, inplace=True)

    train_index = index_shuffled[:int(len(index) * 0.8)]
    test_index = index_shuffled[int(len(index) * 0.8):]

    return train_index, test_index


if __name__ == '__main__':
    net = WeiTopoNet(transpose=TRANSPOSE_DATASET)

    train_index, test_index = get_train_test_indices()

    # Plot distribution of binding affinity in train set
    binding_affinities = train_index['-logKd/Ki'].to_numpy()

    fig = px.histogram(train_index, x='-logKd/Ki', nbins=100, title=f'Training set binding affinity distribution. n={len(train_index)}, range={np.min(binding_affinities)} ~ {np.max(binding_affinities)}')
    fig.write_html('plots/train_set_distribution.html')

    datamodule = ProteinHomologyDataModule(train_index, transpose=TRANSPOSE_DATASET, batch_size=16, homology_base_folder=homologies_base_folder)

    trainer = pl.Trainer(max_epochs=100, accelerator='gpu', devices=1)

    # For training
    trainer.fit(net, datamodule=datamodule)

