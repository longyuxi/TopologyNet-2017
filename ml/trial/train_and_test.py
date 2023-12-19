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

# First train, then test

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..' / '..'/ 'ph'))
sys.path.append(str(Path(__file__).parent / '..' ))

import numpy as np
import pytorch_lightning as pl
from preprocessing import load_pdbbind_data_index
import plotly.express as px
from tqdm import tqdm
import pandas as pd

from models import WeiTopoNet
from dataset import ProteinHomologyDataModule, WeiDataset

###################
# Change these

index_location = '/usr/project/dlab/Users/jaden/pdbbind/index/INDEX_refined_data.2020'
homologies_base_folder = (Path(__file__).parent / '..' / '..' / 'ph'/ 'computed_homologies').resolve()
homologies_base_folder = str(homologies_base_folder)

# See documentation of `WeiTopoNet` in models.py for details on this
TRANSPOSE_DATASET = False
print('transpose dataset:', TRANSPOSE_DATASET)

def get_train_test_indices(train_ratio):
    index = load_pdbbind_data_index(index_location)

    # Randomly train-test split
    index_shuffled = index.iloc[np.random.permutation(len(index))]
    index_shuffled.reset_index(drop=True, inplace=True)

    train_index = index_shuffled[:int(len(index) * train_ratio)]
    test_index = index_shuffled[int(len(index) * train_ratio):]

    return train_index, test_index


# if __name__ == '__main__':
def run(seed, graph_folder, train_ratio, max_epochs=150):
    seed = seed % (2**32)

    graph_folder = Path(graph_folder)
    graph_folder.mkdir(exist_ok=True, parents=True)

    pl.seed_everything(seed)
    net = WeiTopoNet(transpose=TRANSPOSE_DATASET)

    train_index, test_index = get_train_test_indices(train_ratio)

    # Plot distribution of binding affinity in train set
    binding_affinities = train_index['-logKd/Ki'].to_numpy()

    fig = px.histogram(train_index, x='-logKd/Ki', nbins=100, title=f'Training set binding affinity distribution. n={len(train_index)}, range={np.min(binding_affinities)} ~ {np.max(binding_affinities)}')
    fig.write_html(str(graph_folder / 'train_set_distribution.html'))

    datamodule = ProteinHomologyDataModule(train_index, transpose=TRANSPOSE_DATASET, batch_size=16, homology_base_folder=homologies_base_folder)

    trainer = pl.Trainer(max_epochs=max_epochs, accelerator='gpu', devices=1)

    # For training
    trainer.fit(net, datamodule=datamodule)

    test_index.reset_index(drop=True, inplace=True)
    wd = WeiDataset(test_index, transpose=TRANSPOSE_DATASET, return_pdb_code_first=True, homology_base_folder=homologies_base_folder)

    wtn = net  # Same model

    print('Starting predict')

    predicted = []
    actual = []
    pdbcodes = []

    for i in tqdm(range(len(wd))):
        pdbcode, x, y = wd[i]
        y_hat = wtn(x[None, :, :])[0][0].detach().cpu().numpy()
        predicted.append(y_hat)

        actual.append(y[0].detach().cpu().numpy())
        pdbcodes.append(pdbcode)


    predicted = np.array(predicted)
    actual = np.array(actual)

    df = pd.DataFrame(np.stack((predicted, actual), axis=-1), columns=['Predicted -logKd/Ki', 'Actual -logKd/Ki'])

    pearson_corr = df.corr(numeric_only=True).iloc[0, 1]
    mse = np.sum((np.array(predicted) - np.array(actual))**2) / len(predicted)

    print(f'Test set. MSE: {mse:.2f}. Pearson Correlation: {pearson_corr:.2f}')

    # Plot distribution of binding affinity in test set
    binding_affinities = test_index['-logKd/Ki'].to_numpy()
    fig = px.histogram(test_index, x='-logKd/Ki', nbins=100, title=f'Test set binding affinity distribution. n={len(test_index)}, range={np.min(binding_affinities)} ~ {np.max(binding_affinities)}')
    fig.write_html(str(graph_folder / 'test_set_distribution.html'))

    # Plot results
    df = pd.DataFrame(np.stack((predicted, actual), axis=-1), columns=['Predicted -logKd/Ki', 'Actual -logKd/Ki'])
    fig = px.scatter(df, x='Actual -logKd/Ki', y='Predicted -logKd/Ki', title=f'Test set. MSE: {mse:.2f}. Pearson Correlation: {pearson_corr:.2f}')
    fig.write_html(str(graph_folder / 'predictions.html'))

    # Save a version with the pdb codes
    df = pd.DataFrame(np.stack((pdbcodes, predicted, actual), axis=-1), columns=['PDB Code', 'Predicted -logKd/Ki', 'Actual -logKd/Ki'])
    df.to_csv(graph_folder / 'outputs.csv', index=False)

    return {'mse': mse, 'pearson_corr': pearson_corr}

if __name__ == '__main__':
    run(456, 'plots/test', 0.8, 150)