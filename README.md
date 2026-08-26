# Gist VS Details

Trying to understand which KVs can be removed from the KV cache based on the predictive power of memory modules


### Setup

#### Environment
```bash
conda create -n gist_vs_details python=3.12 -y
conda activate gist_vs_details
pip install -r requirements.txt
```

#### .env file
Create a `.env` file in the root directory with the following content:
```
HF_TOKEN="<your_huggingface_token>"
HF_DATASETS_TRUST_REMOTE_CODE=1
TOKENIZERS_PARALLELISM=false
```
