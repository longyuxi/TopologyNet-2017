# Open source implementation of TopologyNet-2017

A reproduction of the TopologyNet-BP algorithm from *TopologyNet: Topology based deep convolutional and multi-task neural networks for biomolecular property predictions* by Zixuan Cang, Guo-Wei Wei. [Paper link](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005690)

This repository is part of *Predicting Affinity Through Homology (PATH): Interpretable Binding Affinity Prediction with Persistent Homology*. The inference code for PATH can be found [in the OSPREY3 package](https://github.com/donaldlab/OSPREY3/tree/main/src/main/python/path). The training code for PATH can be found [here](https://github.com/longyuxi/gbr-tnet).

# Using this repository

*Execution of all the code in this repository is done on an x86 machine running Ubuntu 20*

1. Install [git lfs](https://git-lfs.com/).
2. Clone this repository.

## Core features

1. **Prerequisites**: My software environment is managed with conda.
    1. Install [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/download.html) or [mamba](https://mamba.readthedocs.io/en/latest/mamba-installation.html) (you only need one of the two).
    2. Run `conda env create -f tnet2017.yml` or `mamba env create -f tnet2017.yml` to create the `tnet2017` environment. Activate this environment by `conda activate tnet2017`.
    3. Finally, `pip install -r requirements.txt` to install additional dependencies for this project that are not available through conda.
2. **Construct persistent homology features**: Start with `ph/README.md` to create a feature vector for each protein-ligand complex using persistent homology, by the method described by Cang and Wei.
3. **Neural network**: Start with `ml/README.md` to use the TopologyNet neural network architecture.

## Additional scripts

*I execute these additional scripts in a compute cluster managed by SLURM. To keep track of the statuses of jobs and their results, I use a Redis database and a custom MapReduce-style system.*

I first explain my custom MapReduce-style system. This system consists of two scripts, `job_wrapper.py` and `dispatch_jobs.py`, a SLURM scheduler, and a Redis database. If you are running these scripts in a SLURM cluster, you will need to modify the headers of the temporary shell scripts (see below) to fit the configuration of your cluster. If you are executing these scripts on a compute cluster with a different job scheduler, more changes will need to be made according how compute jobs are submitted on your cluster.

1. Each task is associated with a set of sequentially numbered key starting from a prefix, which is reflected in the `KEY_PREFIX` variable in `dispatch_jobs.py`.
2. `dispatch_jobs.py` will create an entry in the database for each key containing information about the job and the fields {`started`, `finished`, `error`} all set to `False`. It then submits by creating temporary shell scripts that execute `python job_wrapper.py --key {k}` and submit these shell scripts to the SLURM scheduler.
3. `job_wrapper.py` contains the instructions for execution when the work is allocated to a scheduler.

As mentioned, a Redis database is used for managing jobs submitted to the SLURM batch cluster. To set up this database,

1. Build and install Redis via [https://redis.io/docs/getting-started/installation/install-redis-from-source/].
2. Optionally, add the `src` folder of Redis to path.
3. Create a `redis.conf` file somewhere and set a default password by putting e.g. `requirepass topology` in that file.
4. Start the redis server on a host with your `redis.conf` and adjust the `DB` constant in `dispatch_jobs.py` accordingly.

The two additional sets of scripts are:

1. **Perturbation analysis**: Analysis of how much each atom contributes to the prediction of TNet-BP by perturbing them and observing the change in binding affinity predicted by TNet-BP. See `perturbations/README.md`.

2. **Support vector machine (SVM)**: An SVM regressor directly on the feature vector constructed with persistent homology, with feature selection similar to that of PATH (Predicting Affinity Through Homology). See `svm` folder.

*Perturbation analysis came out fruitless. SVM in the style of PATH didn't perform as well as PATH.*


# Cite

See `CITING_OSPREY.txt` for details.

---

*A note on the commit history:* Some commits start with "R:". These commits contain results of intermediate exploratory results.