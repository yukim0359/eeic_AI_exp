import cv2
import sys
import time
import torch
import string
import random
import numpy as np
import torch
import torch.nn as nn
from loss.transformer_Japanese_decomposition import Transformer
import yaml
import os
import argparse
from easydict import EasyDict

from model import tbsrn, tsrn, edsr, srcnn, srresnet, crnn, esrgan

# transformer = Transformer().cuda()
# print(transformer)

# transformer = nn.DataParallel(transformer)
# transformer.load_state_dict(torch.load('./dataset/mydata/pretrain-transformer-stroke-decomposition-chinese.pth'))
# print(transformer)


# # モデルのパラメータをロード
# # state_dict = torch.load('./dataset/mydata/pretrain-transformer-stroke-decomposition-chinese.pth')
# state_dict = torch.load('./dataset/mydata/crnn.pth')
state_dict = torch.load('checkpoint/Japanese_exp/checkpoint.pth')['state_dict_G']

# キー（モデルの層名やパラメータの名前）を表示
# for key in state_dict.keys():
#     if key == 'state_dict_G':
#         for layer in state_dict[key].keys():
#             print(layer)
#             print(state_dict[key][layer].shape)


config_path = os.path.join('config', 'super_resolution.yaml')
config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
config = EasyDict(config)
cfg = config.TRAIN

scale_factor = config.TRAIN.down_sample_scale

model = tsrn.TSRN(scale_factor=scale_factor, width=cfg.width, height=cfg.height,
                              STN=True, mask=False, srb_nums=5, hidden_units=32)
model.load_state_dict(state_dict)

model.eval()
LR_path = 'dataset/original_data/hard/1.png'
SR_path = 'dataset/mydata/result/1.png'

with open(LR_path, 'rb') as f:
    hrBin = f.read()

srBin = model(hrBin)

with open(SR_path, 'wb') as f:
    f.write(srBin)