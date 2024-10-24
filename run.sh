#!/bin/sh
#SBATCH -p v
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# CUDA_VISIBLE_DEVICES=0 python main.py --batch_size=16 --STN --exp_name first_exp --text_focus 
# CUDA_VISIBLE_DEVICES=0 python main.py --batch_size=16 --STN --exp_name first_exp --text_focus --resume YOUR_MODEL --test --test_data_dir ./dataset/mydata/test
# python load_transformer.py
# python save_transformer.py
python challenge_transformer.py