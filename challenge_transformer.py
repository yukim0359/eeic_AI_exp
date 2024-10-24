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

transformer = Transformer().cuda()
transformer.load_state_dict(torch.load('./dataset/mydata/pretrain-transformer-stroke-decomposition-chinese.pth'))
print(transformer)
