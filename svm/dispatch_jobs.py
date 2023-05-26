import glob
import os
import pandas as pd
import pathlib
import logging
import redis
import numpy as np
from Bio.PDB import PDBParser, PDBIO
from tqdm import tqdm
import time
import svm_ph

logging.basicConfig(level=logging.INFO)

###############################
# Platform specific variables #
#                             #
# Change to fit to job        #
###############################


KEY_PREFIX = 'svmph_' # Prefix of every job, as appears in the Redis database
CLUSTER = 'DCC' # 'CS' or 'DCC'

if CLUSTER == 'CS':
    raise NotImplementedError
    cwd = pathlib.Path(__file__).parent.resolve()
    CSV_FILE = f'{cwd}/jobs.csv'
    NUM_JOBS_TO_SUBMIT = 300
    PYTHON_EXECUTABLE = '/usr/project/dlab/Users/jaden/mambaforge/envs/algtop-ph/bin/python'
    ROOT_DIR = '/usr/project/dlab/Users/jaden/algebraic-topology-biomolecules/ph'
    os.system(f'mkdir -p {ROOT_DIR}/slurm-outs')
    SBATCH_TEMPLATE = f"""#!/bin/bash
#SBATCH --time='24:00:00'
#SBATCH --requeue
#SBATCH --chdir={ROOT_DIR}
#SBATCH --output={ROOT_DIR}/slurm-outs/%x-%j-slurm.out
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=compsci

source ~/.zshrc
date
hostname
conda activate algtop-ph
cd {ROOT_DIR}


    """

    DB = redis.Redis(host='login-01', port=6379, decode_responses=True, password="topology")
    FOLDERS = glob.glob('/usr/project/dlab/Users/jaden/pdbbind/refined-set/*')
    PDB_NAMES = [f.split('/')[-1] for f in FOLDERS]

elif CLUSTER == 'DCC':
    NUM_JOBS_TO_SUBMIT = 6000
    PYTHON_EXECUTABLE = '/hpc/group/donald/yl708/mambaforge/envs/tnet2017/bin/python'
    ROOT_DIR = f'{pathlib.Path(__file__).parent.resolve()}'
    SBATCH_TEMPLATE = f"""#!/bin/bash
#SBATCH --partition=common-old,scavenger
#SBATCH --requeue
#SBATCH --chdir={ROOT_DIR}
#SBATCH --output={ROOT_DIR}/slurm_logs/%x-%j-slurm.out
#SBATCH --mem=2500M

source ~/.bashrc
source ~/.bash_profile
date
hostname
conda activate tnet2017
cd {ROOT_DIR}


    """

    DB = redis.Redis(host='dcc-login-03', port=6379, decode_responses=True, password="topology")
    INDEX = svm_ph.INDEX
    BASE_FOLDER = svm_ph.BASE_FOLDER
    OUTPUT_BASE_FOLDER = svm_ph.OUTPUT_BASE_FOLDER

else:
    raise Exception     # Incorrect specification of cluster variable


#############################
# Pre-execution Tests       #
#############################

# Database connection
DB.set('connection-test', '123')
if DB.get('connection-test') == '123':
    DB.delete('abc')
    logging.info('Database connection successful')
else:
    raise Exception     # Database connection failed


#############################
# Actual logic              #
#############################

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

def main(dry_run=False):
    # Initialize database on first run
    if dry_run:
        populate_db()

    # Then submit jobs until either running out of entries or running out of number of jobs to submit
    i = 0

    database_keys = DB.keys(KEY_PREFIX + '*')
    for key in database_keys:
        if i == NUM_JOBS_TO_SUBMIT:
            break
        info = DB.hgetall(key)

        if info['finished'] == 'True' and info['error'] == 'False':
        # if info['attempted'] == 'True':
            continue
        else:
            i += 1
            # submit job for it
            if not dry_run:
                info['attempted'] = 'True'
                DB.hset(key, mapping=info)

                # sbatch run job wrapper
                sbatch_cmd = SBATCH_TEMPLATE + f'\n{PYTHON_EXECUTABLE} {str(pathlib.Path(__file__).parent) + "/job_wrapper.py"} --key {key}'

                # print(sbatch_cmd)
                with open('run.sh', 'w') as f:
                    f.write(sbatch_cmd)

                os.system(f'sbatch --job-name={key} run.sh')

    if dry_run:
        print(f'Number of jobs that would be submitted: {i}')
        time.sleep(5)
    else:
        print(f'Number of jobs submitted: {i}')


def populate_db():
    logging.info('Populating database')

    database_keys = DB.keys()

    update_counter = 0
    for i in tqdm(range(len(INDEX))):
        k = KEY_PREFIX + INDEX.at[i, 'PDB code']

        # if k in database_keys:
        #     logging.debug(f'{k} already in database')
        #     continue

        # Get the index dataframe info as a dictionary
        info = INDEX.iloc[i].to_dict()

        db_entry = {
            'attempted': 'False',
            'error': 'False',
            'finished': 'False',
            'protein_file': f'{BASE_FOLDER}/{info["PDB code"]}/{info["PDB code"]}_protein.pdb',
            'ligand_file': f'{BASE_FOLDER}/{info["PDB code"]}/{info["PDB code"]}_ligand.mol2',
            'save_folder': str(OUTPUT_BASE_FOLDER),
            **info
        }


        DB.hset(k, mapping=db_entry)
        update_counter += 1

    logging.info(f'Updated {update_counter} entries in database')

def rebuild_db():
    raise NotImplementedError

def get_db():
    # Pinnacle of OOP
    return DB

if __name__ == '__main__':
    # rebuild_db()
    main(dry_run=True)
    main(dry_run=False)
