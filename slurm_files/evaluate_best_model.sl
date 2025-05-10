#!/bin/bash
#SBATCH --job-name=cpu_eval            # Job name
#SBATCH --output=output/%x_%j.txt             # Output file
#SBATCH --error=error/%x_%j.txt               # Error file
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1 --cpus-per-task=32     # 32 CPU on a single node
#SBATCH --mem-per-cpu=20g                 # Memory request per CPU
#SBATCH --partition=short              # >12h partition
#SBATCH --time=02:00:00                  # Time limit (hrs:min:sec)
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=grantcr@bc.edu # Email for notifications

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
python league.py\
    --evals /home/grantcr/MicroRTS-Py_Tree_Search/experiments/models/MicroRTSGridModeVecEnv__ppo_gridnet_tree__1746664574__1746664574/14825472.pt\
    --update-db True\
    --cuda False\
    --output_path /home/grantcr/MicroRTS-Py_Tree_Search/experiments/14825472_eval.csv\
    --prod-mode True\
    --num_matches 100
