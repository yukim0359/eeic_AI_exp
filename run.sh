#!/bin/sh
#SBATCH -p v
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0 python main.py --batch_size=32 --STN --exp_name Japanese_exp_6 --text_focus 
# CUDA_VISIBLE_DEVICES=0 python main.py --batch_size=32 --STN --exp_name Japanese_exp --text_focus --resume checkpoint/Japanese_exp/checkpoint.pth --test --test_data_dir ./dataset/mydata/test
# python load_TSRN.py
# python save_transformer.py
# python challenge_transformer.py