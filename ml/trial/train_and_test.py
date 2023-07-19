# First train, then test

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..' / '..'/ 'ph'))
sys.path.append(str(Path(__file__).parent / '..' ))
import argparse

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from preprocessing import load_pdbbind_data_index
import plotly.express as px
from tqdm import tqdm
import pandas as pd

from models import SusNet, WeiTopoNet, MLPTopoNet, AttentionTopoNet
from dataset import ProteinHomologyDataModule, WeiDataset

###################
# Change these

index_location = '/home/longyuxi/Documents/mount/pdbbind-dataset/index/INDEX_refined_data.2020'
homologies_base_folder = (Path(__file__).parent / '..' / '..' / 'ph'/ 'computed_homologies').resolve()
homologies_base_folder = str(homologies_base_folder)

# See documentation of `WeiTopoNet` in models.py for details on this
TRANSPOSE_DATASET = False

##################


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


    pl.seed_everything(123)
    net = WeiTopoNet(transpose=TRANSPOSE_DATASET)

    train_index, test_index = get_train_test_indices()

    # Plot distribution of binding affinity in train set
    binding_affinities = train_index['-logKd/Ki'].to_numpy()

    fig = px.histogram(train_index, x='-logKd/Ki', nbins=100, title=f'Training set binding affinity distribution. n={len(train_index)}, range={np.min(binding_affinities)} ~ {np.max(binding_affinities)}')
    fig.write_html('plots/train_set_distribution.html')

    datamodule = ProteinHomologyDataModule(train_index, transpose=TRANSPOSE_DATASET, batch_size=16, homology_base_folder=homologies_base_folder)

    # trainer = pl.Trainer(max_epochs=100, accelerator='gpu', devices=1)
    trainer = pl.Trainer(max_epochs=100, accelerator='gpu', devices=1)  # For testing

    # For training
    trainer.fit(net, datamodule=datamodule)


    test_index.reset_index(drop=True, inplace=True)
    wd = WeiDataset(test_index, transpose=TRANSPOSE_DATASET, return_pdb_code_first=True, homology_base_folder=homologies_base_folder)

    wtn = net  # Same model

    print('Starting predict')

    predicted = []
    actual = []
    pdbcodes = []

    peaked_molecules = []
    regular_molecules = []

    for i in tqdm(range(len(wd))):
        pdbcode, x, y = wd[i]
        y_hat = wtn(x[None, :, :])[0][0].detach().cpu().numpy()
        predicted.append(y_hat)

        actual.append(y[0].detach().cpu().numpy())
        pdbcodes.append(pdbcode)


    predicted = np.array(predicted)
    actual = np.array(actual)

    save_base_folder = Path('plots')
    save_base_folder.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(np.stack((predicted, actual), axis=-1), columns=['Predicted -logKd/Ki', 'Actual -logKd/Ki'])

    pearson_corr = df.corr(numeric_only=True).iloc[0, 1]
    mse = np.sum((np.array(predicted) - np.array(actual))**2) / len(predicted)

    print(f'Test set. MSE: {mse:.2f}. Pearson Correlation: {pearson_corr:.2f}')

    # Plot distribution of binding affinity in test set
    binding_affinities = test_index['-logKd/Ki'].to_numpy()
    fig = px.histogram(test_index, x='-logKd/Ki', nbins=100, title=f'Test set binding affinity distribution. n={len(test_index)}, range={np.min(binding_affinities)} ~ {np.max(binding_affinities)}')
    fig.write_html(str(save_base_folder / 'test_set_distribution.html'))

    # Plot results
    df = pd.DataFrame(np.stack((predicted, actual), axis=-1), columns=['Predicted -logKd/Ki', 'Actual -logKd/Ki'])
    fig = px.scatter(df, x='Actual -logKd/Ki', y='Predicted -logKd/Ki', title=f'Test set. MSE: {mse:.2f}. Pearson Correlation: {pearson_corr:.2f}')
    fig.write_html(str(save_base_folder / 'predictions.html'))

    # Save a version with the pdb codes
    df = pd.DataFrame(np.stack((pdbcodes, predicted, actual), axis=-1), columns=['PDB Code', 'Predicted -logKd/Ki', 'Actual -logKd/Ki'])
    df.to_csv(save_base_folder / 'outputs.csv', index=False)