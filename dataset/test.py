import os
import lmdb  # install lmdb by "pip install lmdb"
import cv2
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import scipy.io as sio
from scipy.io import loadmat
from tqdm import tqdm
import re, time
import random
import six

with open("data/label/1.txt") as f:
    label = f.read().strip()
print(repr(label))