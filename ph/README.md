# Procedure for creating the feature set:

The following procedure eventually creates a `cmds.tmp` file. Each line of this file is an independent command to be run and corresponds to computing the persistent homology of a particular protein-ligand complex.

1. Pre-requisites: None. If you wish to run these commands in parallel on a multi-core computer, you may consider [GNU Parallel](https://www.gnu.org/software/parallel/).

2. Get PDBBind dataset
    ```bash
    cd [some_folder]
    bash download_pdbbind_and_clean.sh
    ```
3. Modify the parameters on the top of `make-ph-cmds.py` accordingly.

4. Run `python make-ph-cmds.py` to create `cmds.tmp`. Then run somethig like `time parallel -j 10 --eta < cmds.tmp` to execute the commands in parallel with GNU Parallel. Modify this procedure accordingly for distributed computing.

