#!/usr/bin/bash
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
echo "Run 'conda init' later to make sure is loaded properly in bash shells"

conda create -n pem-llm python=3.12 -y
conda activate pem-llm
pip install -r requirements.txt

# enable write access in mounted nfs - user only
sudo chown -R ma-user:ma-group /home/ma-user/work

# enable write access in mounted nfs - full group
# sudo chmod -R g+rwX /home/ma-user/work

# install git
sudo apt-get update
sudo apt install -y git-all
echo "Now that git is installed, add your git credentials+PAT to ~/.netrc"

# point to HF_HOME
grep -qxF 'export HF_HOME="/home/ma-user/.cache/huggingface"' ~/.bashrc || \
  echo 'export HF_HOME="/home/ma-user/.cache/huggingface"' >> ~/.bashrc
