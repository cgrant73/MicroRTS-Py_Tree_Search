<p align="center">
    <img src="https://raw.githubusercontent.com/Farama-Foundation/MicroRTS-Py/master/micrortspy-text.png" width="500px"/>
</p>

Tree Search Algorithm in MicroRTS-Py.

[<img src="https://img.shields.io/badge/discord-gym%20microrts-green?label=Discord&logo=discord&logoColor=ffffff&labelColor=7289DA&color=2c2f33">](https://discord.gg/DdJsrdry6F)
[<img src="https://github.com/vwxyzjn/gym-microrts/workflows/build/badge.svg">](https://github.com/Farama-Foundation/MicroRTS-Py/actions)
[<img src="https://badge.fury.io/py/gym-microrts.svg">](
https://pypi.org/project/gym-microrts/)

This repo contains the source code for the gym wrapper of μRTS authored by [Santiago Ontañón](https://github.com/santiontanon/microrts).

MicroRTS-Py will eventually be updated, maintained, and made compliant with the standards of the Farama Foundation (https://farama.org/project_standards). However, this is currently a lower priority than other projects we're working to maintain. If you'd like to contribute to development, you can join our discord server here- https://discord.gg/jfERDCSw.

![demo.gif](static/fullgame.gif)

## Get Started

Prerequisites:
* Python 3.8+
* [Poetry](https://python-poetry.org)
* Java 8.0+
* FFmpeg (for video recording utilities)

```bash
$ git clone --recursive https://github.com/cgrant73/MicroRTS-Py_Tree_Search.git && \
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh 
```
Follow Prompts from Miniconda. Then:
```bash
bash
source ~/.bashrc
pipx install poetry
source ~/miniconda3/bin/activate
conda create -n "urtsenv" python=3.9 ipython
conda activate urtsenv
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
poetry self add poetry-plugin-shell
cd MicroRTS-Py_Tree_Search
poetry install
# The `poetry install` command above creates a virtual environment for us, in which all the dependencies are installed.
# We can use `poetry shell` to create a new shell in which this environment is activated. Once we are done working with
# MicroRTS, we can leave it again using `exit`.
poetry shell
# By default, the torch wheel is built with CUDA 10.2. If you are using newer NVIDIA GPUs (e.g., 3060 TI), you may need to specifically install CUDA 11.3 wheels by overriding the torch dependency with pip:
# poetry run pip install "torch==1.12.1" --upgrade --extra-index-url https://download.pytorch.org/whl/cu113
python hello_world.py
```

If the `poetry install` command gets stuck on a Linux machine, [it may help to first run](https://github.com/python-poetry/poetry/issues/8623): `export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`.

To train an agent, run the following

```bash
cd experiments
python ppo_gridnet.py \
    --total-timesteps 100000000 \
    --capture-video \
    --seed 1
```

[![asciicast](https://asciinema.org/a/586754.svg)](https://asciinema.org/a/586754)

For running a partial observable example, tune the `partial_obs` argument.
```bash
cd experiments
python ppo_gridnet.py \
    --partial-obs \
    --capture-video \
    --seed 1
```