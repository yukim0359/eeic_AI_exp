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

# transformer = Transformer().cuda()
# print(transformer)

# transformer = nn.DataParallel(transformer)
# transformer.load_state_dict(torch.load('./dataset/mydata/pretrain-transformer-stroke-decomposition-chinese.pth'))
# print(transformer)


# # モデルのパラメータをロード
# # state_dict = torch.load('./dataset/mydata/pretrain-transformer-stroke-decomposition-chinese.pth')
# state_dict = torch.load('./dataset/mydata/crnn.pth')
state_dict = torch.load('test.pth')

# キー（モデルの層名やパラメータの名前）を表示
for key in state_dict.keys():
    print(key)
    print(state_dict[key].shape)
