import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '..'/ 'ph'))

import argparse
import pandas as pd
import traceback
import dispatch_jobs
import calculate_2345_homology
import calculate_pairwise_opposition_homologies_binned

DB = dispatch_jobs.get_db()

def job(key):
    # The part where the job actually runs, given df and idx as input

    # Each entry in the redis database should be a dictionary in the following form

    # job_index (key_prefix + job_index = key in database),
    # {
    #     protein_file: name of pdb file,
    #     ligand_file: name of ligand mol2 file,
    #     save_folder: where to save the output
    #     atom_index: index of atom that is perturbed,
    #     perturbation_index: perturbation index (passed in to the perturbation function)
    #     attempted: true/false
    #     error: true/false
    #     finished: true/false
    # }

    d = DB.hgetall(key)

    calculate_2345_homology.run(d['protein_file'], d['ligand_file'], d['save_folder'] + '/2345_homology.pckl')
    calculate_pairwise_opposition_homologies_binned.run(d['protein_file'], d['ligand_file'], d['save_folder'] + '/pairwise_opposition.pckl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--key')

    args = parser.parse_args()
    key = args.key

    print('key', key)

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
