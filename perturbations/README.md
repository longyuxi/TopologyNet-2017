### Pre-requisites

I use [mamba](https://mamba.readthedocs.io/en/latest/installation.html) instead of conda. If you use conda, swap out all the `mamba` commands into `conda`.

1. `mamba create -n tnet2017 pandas redis-py biopython numpy biopandas python=3.10`
2. `mamba activate tnet2017`
3. `pip install -U giotto-tda --user` (or install from wheel. Note that using python=3.10 above allows us to directly use a pre-built wheel.)

**Starting a Redis database**: The Redis database is used for managing jobs submitted to the SLURM batch cluster.
1. Make and install Redis via [https://redis.io/docs/getting-started/installation/install-redis-from-source/].
2. Optionally, add the `src` folder of Redis to path.
3. Create a `redis.conf` file somewhere and set a default password by putting e.g. `requirepass topology` in that file.
4. Start the redis server on a host and adjust the `DB` constant in `dispatch_jobs.py` accordingly.

