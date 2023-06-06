import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'/ 'ph'))

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from models import SusNet, WeiTopoNet, MLPTopoNet, AttentionTopoNet
from preprocessing import load_pdbbind_data_index
from dataset import ProteinHomologyDataModule, WeiDataset

###################
# Change these

index_location = '/home/longyuxi/Documents/mount/pdbbind-dataset/index/INDEX_refined_data.2020'
homologies_base_folder = (Path(__file__).parent / '..' / 'ph'/ 'computed_homologies').resolve()
homologies_base_folder = str(homologies_base_folder)

# See documentation of `WeiTopoNet` in models.py for details on this
TRANSPOSE_DATASET = True

##################

pl.seed_everything(42)

print('transpose dataset:', TRANSPOSE_DATASET)

def get_train_test_indices():
    index = load_pdbbind_data_index(index_location)

    # Randomly train-test split
    index_shuffled = index.iloc[np.random.permutation(len(index))]
    index_shuffled.reset_index(drop=True, inplace=True)

    train_index = index[:int(len(index) * 0.8)]
    test_index = index[int(len(index) * 0.8):]

    return train_index, test_index


if __name__ == '__main__':
    net = WeiTopoNet(transpose=TRANSPOSE_DATASET)

    train_index, test_index = get_train_test_indices()
    datamodule = ProteinHomologyDataModule(train_index, transpose=TRANSPOSE_DATASET, batch_size=16, homology_base_folder=homologies_base_folder)

    trainer = pl.Trainer(max_epochs=2000, accelerator='gpu', devices=1)

    # For training
    trainer.fit(net, datamodule=datamodule)

