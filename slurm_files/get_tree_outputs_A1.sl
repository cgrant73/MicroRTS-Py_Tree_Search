#!/bin/bash
#SBATCH --job-name=tree_outputs           # Job name
#SBATCH --output=output/%x_%j.txt             # Output file
#SBATCH --error=error/%x_%j.txt               # Error file
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1 --cpus-per-task=32     # 32 CPU on a single node
#SBATCH --mem-per-cpu=5g                 # Memory request per CPU
#SBATCH --time=10:00:00                  # Time limit (hrs:min:sec)
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=grantcr@bc.edu # Email for notifications

module load cuda
cd /mmfs1/data/grantcr/
bash 
cd /mmfs1/data/grantcr/MicroRTS-Py_Tree_Search/experiments
conda activate urtsenv
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
poetry install
poetry build
poetry run pip install "torch==1.12.1" --upgrade --extra-index-url https://download.pytorch.org/whl/cu113
source ~/miniconda3/bin/activate
conda activate urtsenv
poetry shell
conda activate urtsenv
poetry shell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
python cacheSamples.py
python cacheMyKingdom.py