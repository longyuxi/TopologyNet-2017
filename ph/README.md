# Procedure for creating the dataset:

The following procedure eventually creates a `cmds.tmp` file. Each line of this file is an independent command to be run.

1. Pre-requisites:
    **TODO**

2. Get PDBBind dataset
    ```bash
    cd [some_folder]
    bash download_pdbbind_and_clean.sh
    ```
3. Modify the parameters on the top of `make-ph-cmds.py` accordingly.

4. Run `python make-ph-cmds.py` to create `cmds.tmp`. Then run somethig like `time parallel -j 10 --eta < cmds.tmp` to execute the commands in parallel on your own machine. Modify this procedure accordingly for distributed computing.

