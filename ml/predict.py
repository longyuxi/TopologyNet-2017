# %%
# For evaluating model output. This script outputs into the 'plots' directory.

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from preprocessing import load_pdbbind_data_index
import seaborn as sns
import glob
from tqdm import tqdm

from dataset import WeiDataset
from models import WeiTopoNet


if sys.platform == 'linux':
    index_location = '/home/longyuxi/Documents/mount/pdbbind-dataset/index/INDEX_refined_data.2020'
else:
    raise Exception


TRANSPOSE_DATASET = True # Set this to whatever is done at training time

index = load_pdbbind_data_index(index_location)
wd = WeiDataset(index, transpose=TRANSPOSE_DATASET, return_pdb_code_first=True)
wtn = WeiTopoNet()
wtn = wtn.load_from_checkpoint(glob.glob('lightning_logs/transpose_glorot/checkpoints/*')[0], transpose=TRANSPOSE_DATASET)


predicted = []
actual = []
pdbcodes = []

peaked_molecules = []
regular_molecules = []

for i in tqdm(range(5000)):
    pdbcode, x, y = wd[i]
    y_hat = wtn(x[None, :, :])[0][0].detach().cpu().numpy()
    predicted.append(y_hat)

    actual.append(y[0].detach().cpu().numpy())
    pdbcodes.append(pdbcode)

    if str(y_hat) == '6.413805':
        peaked_molecules.append(x)
    else:
        regular_molecules.append(x)



predicted = np.array(predicted)
actual = np.array(actual)

save_base_folder = Path('plots')
save_base_folder.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(np.stack((predicted, actual), axis=-1), columns=['Predicted -logKd/Ki', 'Actual -logKd/Ki'])

# %%

pearson_corr = df.corr(numeric_only=True)
mse = np.sum((np.array(predicted) - np.array(actual))**2) / len(predicted)

plt.clf()
fig = plt.figure()
sns.scatterplot(data=df, x='Actual -logKd/Ki', y='Predicted -logKd/Ki')
ax = fig.gca()
ax.set_title(f'MSE: {mse}. $R_p$: {pearson_corr}')
imfile = save_base_folder / 'predictions.jpg'
plt.savefig(imfile)

# Save a version with the pdb codes
df = pd.DataFrame(np.stack((pdbcodes, predicted, actual), axis=-1), columns=['PDB Code', 'Predicted -logKd/Ki', 'Actual -logKd/Ki'])
df.to_csv(save_base_folder / 'outputs.csv', index=False)