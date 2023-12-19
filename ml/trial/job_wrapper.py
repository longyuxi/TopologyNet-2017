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
sys.path.append(str(Path(__file__).parent / '..'))

import argparse
import traceback

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

    results = train_and_test.run(seed, graph_folder, train_ratio, max_epochs=max_epochs)

    # Save results
    DB.hset(key, mapping={**d, **results})



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
