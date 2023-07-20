import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'))

import argparse
import traceback
from gtda.diagrams import PersistenceImage
import numpy as np
from tqdm import tqdm

import dispatch_jobs
import train_and_test
import torch

DB = dispatch_jobs.get_db()


def job(key):
    # The part where the job actually runs
    # {'train_ratio': '0.9', 'graph_folder': '/usr/project/dlab/Users/jaden/tnet2017-new/ml/trial/plots/tnet2017_trial_0', 'seed': '-8745064292264974360', 'finished': 'False', 'max_epochs': '150', 'error': 'False', 'attempted': 'False'}


    d = DB.hgetall(key)
    print(f'GPU is avilable? {torch.cuda.is_available()}')
    print(d)

    seed = int(d['seed']) % (2**32)
    train_ratio = float(d['train_ratio'])
    graph_folder = d['graph_folder']
    max_epochs = int(d['max_epochs'])

    train_and_test.run(seed, graph_folder, train_ratio, max_epochs=max_epochs)



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
