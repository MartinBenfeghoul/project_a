#/bin/bash -e

conda create -n pem-llm python=3.12 -y
conda activate pem-llm
pip install -r requirements.txt
