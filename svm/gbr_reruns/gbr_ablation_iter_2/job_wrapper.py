import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

import argparse
import traceback
import pickle
from gtda.diagrams import PersistenceImage
import numpy as np

import dispatch_jobs
import analyze_gbr_importance

DB = dispatch_jobs.get_db()
pim = PersistenceImage()

def job(key):
    # The part where the job actually runs, given df and idx as input

    d = DB.hgetall(key)

    print(d)

    seed = int(d['seed']) % (2**32)
    train_ratio = float(d['train_ratio'])

    import numpy as np
    from pathlib import Path
    import pickle

    # Loading the files back in
    observations = []
    binding_affinities = []

    NPYS_FOLDER = Path('/usr/project/dlab/Users/jaden/tnet2017-new/svm/npys')

    for file_index in range(len(list(NPYS_FOLDER.glob('binding_affinities_*.npy')))):
        with open(NPYS_FOLDER / f'observations_{file_index}.npy', 'rb') as f:
            observations.append(np.load(f))

        with open(NPYS_FOLDER / f'binding_affinities_{file_index}.npy', 'rb') as f:
            binding_affinities.append(np.load(f))

    observations = np.concatenate(observations)
    binding_affinities = np.concatenate(binding_affinities)

    if 'test' in d:
        print('Test key detected, using only 10 samples')
        observations = observations[:10]
        binding_affinities = binding_affinities[:10]

    important_feature_indices = [11211, 11918, 11110, 10605, 12120, 879802, 281716, 10807, 10504, 281211, 11413, 281817, 282120, 1120119, 11716, 280908, 12221, 282221, 879902, 38110, 1312, 11514, 10706, 11817, 11109, 9900]

    important_diagrams = [int(i / 10000) for i in important_feature_indices]
    important_diagrams = np.unique(important_diagrams)

    observations = observations.reshape(observations.shape[0], 120, 10000)
    selected_observations = observations[:, important_diagrams, :]
    selected_observations = selected_observations.reshape(selected_observations.shape[0], -1)  # Flatten out for SVM


    highly_selected_observations_indices = [100, 10504, 10605, 10706, 10807, 11009, 11110, 11211, 11312, 11413, 11817, 11918, 12019, 12120, 12221, 13332, 13635,  1501, 24500, 25900, 28100, 28800, 28900, 29700, 29800, 30201, 30302, 30908, 31110, 31211, 31716, 31817, 32221, 40002, 40100, 40102, 40300, 47903, 48102, 49700, 49701, 49702, 49802, 49900, 49902, 50225, 50526, 54096, 56001, 57026, 57201, 57525, 58423, 58432, 58607, 58830, 58831, 58930, 59202, 59205, 59932,  9500,  9900]

    highly_selected_observations = selected_observations[:, highly_selected_observations_indices]

    del observations
    del selected_observations

    from sklearn.model_selection import train_test_split


    print(highly_selected_observations.shape)

    X_train, X_test, y_train, y_test = train_test_split(highly_selected_observations, binding_affinities, train_size=train_ratio, random_state=seed)

    from sklearn.ensemble import GradientBoostingRegressor

    regr = GradientBoostingRegressor()
    print('Starting fit')

    regr.fit(X_train, y_train)

    import plotly.express as px
    from sklearn.metrics import r2_score

    print('Starting predict')

    y_hat = regr.predict(X_test)

    fig = px.scatter(x=y_test, y=y_hat, labels={'x': 'True binding affinity', 'y': 'Predicted binding affinity'}, title=f'True vs Predicted binding affinity\nR^2 = {r2_score(y_test, y_hat):.2f}, n = {len(y_test)}, RMSE: {np.sqrt(np.mean((y_test - y_hat)**2)):.2f}, Pearson: {np.corrcoef(y_test, y_hat)[0, 1]:.2f}')
    fig.write_html(d['scatter_plot_file'])

    np.save(d['prediction_results_file'], np.stack((y_test, y_hat)))

    print(f'RMSE: {np.sqrt(np.mean((y_test - y_hat) ** 2)): .4f}')
    print(f'Pearson correlation: {np.corrcoef(y_test, y_hat)[0, 1]: .4f}')

    d['r2'] = f'{r2_score(y_test, y_hat):.4f}'
    d['rmse'] = f'{np.sqrt(np.mean((y_test - y_hat) ** 2)):.4f}'
    d['pearson'] = f'{np.corrcoef(y_test, y_hat)[0, 1]:.4f}'
    DB.hset(key, mapping=d)

    analyze_gbr_importance.analyze(regr, seed=seed, impurity_importance_html=d['impurity_importance_html'], sorted_impurity_importances_pckl=d['sorted_impurity_importances_pckl'], permutation_importance_html=d['permutation_importance_html'], sorted_permutation_importances_pckl=d['sorted_permutation_importances_pckl'], test_observations=X_test, test_binding_affinities=y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--key')

    args = parser.parse_args()
    key = args.key

    print('key', key)
    print('job started')
    d = DB.hgetall(key)
    d['attempted'] = 'True'
    DB.hset(key, mapping=d)

    try:
        job(key)
        print('job finished')
        d = DB.hgetall(key)
        d['finished'] = 'True'
        d['error'] = 'False'
        DB.hset(key, mapping=d)
        print('job success')

    except Exception as err:
        print(Exception, err)
        print(traceback.format_exc())
        print('job error')

        d = DB.hgetall(key)
        d['finished'] = 'True'
        d['error'] = 'True'
        DB.hset(key, mapping=d)
