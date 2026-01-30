#/bin/bash -e

conda create -n pem-llm python=3.12 -y
conda activate pem-llm
pip install -r requirements.txt

# enable write access in mounted nfs - user only
# sudo chown -R ma-user:ma-group /home/ma-user/work

# enable write access in mounted nfs - full group
sudo chmod -R g+rwX /home/ma-user/work

# point to HF_HOME
export HF_HOME="/home/ma-user/.cache/huggingface"