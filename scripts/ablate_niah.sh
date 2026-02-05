#/bin/bash

seq_lens="1k,2k,4k,8k"
comp_ratios="1.5,2.0,2.5,3.0,3.5,4.0"
n_samples=100

# Baseline
python ablate_niah.py -c baseline -n "$n_samples" --seq_lens "$seq_lens" --comp_ratios "$comp_ratios"

# SVD
python ablate_niah.py -c svd -n "$n_samples" --seq_lens "$seq_lens" --comp_ratios "$comp_ratios"

# Surprise SVD
python ablate_niah.py -c surprise_svd -n "$n_samples" --seq_lens "$seq_lens" --comp_ratios "$comp_ratios"
