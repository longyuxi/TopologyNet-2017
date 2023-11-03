# Open source implementation of TopologyNet-2017

A reproduction of the TopologyNet-BP algorithm from *TopologyNet: Topology based deep convolutional and multi-task neural networks for biomolecular property predictions* by Zixuan Cang, Guo-Wei Wei. [Paper link](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005690)

# Using this repository

*Execution of all the code in this repository is done on an x86 machine running Ubuntu 20*

1. **Prerequisites**: I manage my software environment with conda. Install [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/download.html) or [mamba](https://mamba.readthedocs.io/en/latest/mamba-installation.html) (you only need one of the two). Then run `conda env create -f tnet2017.yml` or `mamba env create -f tnet2017.yml` to create the `tnet2017` environment. Activate this environment by `conda activate tnet2017`. Finally, `pip install -r requirements.txt` to install additional dependencies for this project that are not available through conda.
2. **Construct persistent homology features**: Start with `ph/README.md` to calculate the persistent homology features.
3. **Neural network**: Start with `ml/README.md` to use the TopologyNet neural network architecture.


---

*A note on the commit history:* Some commits start with "R:". These commits contain results of intermediate exploratory results.