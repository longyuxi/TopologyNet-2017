# Open source implementation of TopologyNet-2017

A reproduction of the TopologyNet-BP algorithm from *TopologyNet: Topology based deep convolutional and multi-task neural networks for biomolecular property predictions* by Zixuan Cang, Guo-Wei Wei. [Paper link](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005690)

# Using this repository

*Execution of all the code in this repository is done on an x86 machine running Ubuntu 20*

## Core features

1. **Prerequisites**: I manage my software environment with conda.
    1. Install [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/download.html) or [mamba](https://mamba.readthedocs.io/en/latest/mamba-installation.html) (you only need one of the two).
    2. Run `conda env create -f tnet2017.yml` or `mamba env create -f tnet2017.yml` to create the `tnet2017` environment. Activate this environment by `conda activate tnet2017`.
    3. Finally, `pip install -r requirements.txt` to install additional dependencies for this project that are not available through conda.
2. **Construct persistent homology features**: Start with `ph/README.md` to create a feature vector for each protein-ligand complex using persistent homology, by the method described by Cang and Wei.
3. **Neural network**: Start with `ml/README.md` to use the TopologyNet neural network architecture.

## Additional scripts

*These additional scripts take a long time to I in a server cluster managed by SLURM. To keep track of the statuses of jobs and their results, I use a Redis database and a custom MapReduce-style system.*

I first explain my custom MapReduce-style system. This system consists of two scripts, `job_wrapper.py` and `dispatch_jobs.py`, a SLURM scheduler, and a Redis database. If these scripts are being executed on a different job scheduler, they will need to be adjusted accordingly.

1. Each task is associated with a set of sequentially numbered key starting from a prefix, which is reflected in the `KEY_PREFIX` variable in `dispatch_jobs.py`.
2. `dispatch_jobs.py` will create an entry in the database for each key containing information about the job and the fields {`started`, `finished`, `error`} all set to `False`. It then submits by creating temporary shell scripts that execute `python job_wrapper.py --key {k}` and submit these shell scripts to the SLURM scheduler.
3. `job_wrapper.py` contains the instructions for execution when the work is allocated to a scheduler.

Instructions for running these additional scripts are then:

1. **Setting up the database**: The Redis database is used for managing jobs submitted to the SLURM batch cluster.
    1. Build and install Redis via [https://redis.io/docs/getting-started/installation/install-redis-from-source/].
    2. Optionally, add the `src` folder of Redis to path.
    3. Create a `redis.conf` file somewhere and set a default password by putting e.g. `requirepass topology` in that file.
    4. Start the redis server on a host and adjust the `DB` constant in `dispatch_jobs.py` accordingly.
2. **Perturbation analysis**: Analysis of how much each atom contributes to the prediction of TNet-BP by perturbing them. See `perturbations/README.md`

*A script requiring further documentation:*

3. **Support vector machine (SVM)**: An SVM regressor directly on the feature vector constructed with persistent homology, with feature curation similar to that of PATH (Predicting Affinity Through Homology).


---

*A note on the commit history:* Some commits start with "R:". These commits contain results of intermediate exploratory results.