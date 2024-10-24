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
torch.save(transformer.state_dict(), 'test.pth')
print('OK')