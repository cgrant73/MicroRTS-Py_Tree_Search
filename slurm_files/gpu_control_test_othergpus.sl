#!/bin/bash
#SBATCH --job-name=gpu_control_test            # Job name
#SBATCH --output=output/%x_%j.txt             # Output file
#SBATCH --error=error/%x_%j.txt               # Error file
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1 --cpus-per-task=10     # 10 CPU on a single node
#SBATCH --mem-per-cpu=20g                 # Memory request per CPU
#SBATCH --partition=short              # >12h partition
#SBATCH --time=00:05:00                  # Time limit (hrs:min:sec)
#SBATCH --gres=gpu:1			# one GPU
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=grantcr@bc.edu # Email for notifications
#SBATCH --exclude="g002"		# Fails for g002, cannot allow it to be chosen

module load cuda
cd /home/grantcr/
bash 
cd /home/grantcr/MicroRTS-Py_Tree_Search/experiments/
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
python ppo_gridnet_tree.py\
    --seed 0\
    --prod-mode True\
    --num-bot-envs 24\
    --num-selfplay-envs 4\
    --cuda True