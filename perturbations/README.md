### Pre-requisites

I use [mamba](https://mamba.readthedocs.io/en/latest/installation.html) instead of conda. If you use conda, swap out all the `mamba` commands into `conda`.

<!-- *Note that this does not install PyTorch. The packages required to run the machine learning scripts should be installed in a separate environment.* -->

Find your installation instruction of PyTorch on the [official website](https://pytorch.org/get-started/locally/) based on your CUDA version. So for me, the command is

```
mamba create -n tnet2017 pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```

Then install the other packages:

```
mamba activate tnet2017
mamba install pandas redis-py biopython numpy biopandas seaborn tqdm pytorch-lightning
pip install -U giotto-tda --user
```

(or install from wheel. Note that using python=3.10 above allows us to directly use a pre-built wheel.)

**Starting a Redis database**: The Redis database is used for managing jobs submitted to the SLURM batch cluster.
1. Make and install Redis via [https://redis.io/docs/getting-started/installation/install-redis-from-source/].
2. Optionally, add the `src` folder of Redis to path.
3. Create a `redis.conf` file somewhere and set a default password by putting e.g. `requirepass topology` in that file.
4. Start the redis server on a host and adjust the `DB` constant in `dispatch_jobs.py` accordingly.

