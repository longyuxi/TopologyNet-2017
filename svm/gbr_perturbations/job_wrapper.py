import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

import argparse
import traceback
import pickle
from gtda.diagrams import PersistenceImage
import numpy as np

import dispatch_jobs
import svm_ph

DB = dispatch_jobs.get_db()
pim = PersistenceImage()

def job(key):
    # The part where the job actually runs, given df and idx as input

    # Each entry in the redis database should be a dictionary in the following form
    # TODO
    # {'save_folder': '/usr/project/dlab/Users/jaden/perturbations/gbr_perturb_1a1e_9', 'error': 'False', 'finished': 'True', 'protein_file': '/usr/project/dlab/Users/jaden/perturbations/gbr_perturb_1a1e_9/gbr_perturb_1a1e_9.pdb', 'attempted': 'True', 'ligand_file': '/usr/project/dlab/Users/jaden/perturbations/1a1e_ligand.mol2', 'perturbation_index': '3', 'atom_index': '1'}

    d = DB.hgetall(key)
    print(f'Job ID: {key}\nData: {d}')

    # Calculate the persistence diagrams
    protein_file = d['protein_file']
    ligand_file = d['ligand_file']

    pw_opposition_diagrams = svm_ph.get_persistence_diagrams(protein_file, ligand_file)
    other_persistence_diagrams = svm_ph.get_2345_persistence_diagrams(protein_file, ligand_file)

    data_dict = {
            'pw_opposition_diagrams': pw_opposition_diagrams,
            'other_persistence_diagrams': other_persistence_diagrams
            }
    
    all_diagrams = pw_opposition_diagrams + other_persistence_diagrams
    all_images = list(map(lambda x: pim.fit_transform(x), all_diagrams))
    all_images = np.array(all_images).flatten().reshape(1, -1)

    # Load GBR model
    model = '/usr/project/dlab/Users/jaden/tnet2017-new/svm/GBR_model.pckl'
    with open(model, 'rb') as f:
        gbr = pickle.load(f)

    y_hat = gbr.predict(all_images)[0]

    d['y_hat'] = str(y_hat)
    DB.hset(key, mapping=d)


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
