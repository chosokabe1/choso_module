from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import sys

def makedir(dir):
  if not os.path.exists(dir):
    os.makedirs(dir)

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "C:\ex\sen\data\itoteki\cross1"
# data_dir = "C:\ex\sen\data\max_square\\train_val"
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "vit_b_32"
#結果出力ディレクトリ
output_name = "vit_b_16_itoteki_cross1"
# Number of classes in the dataset
num_classes = 2
# Batch size for training (change depending on how much memory you have)
batch_size = 32
# Number of epochs to train for
num_epochs = 60
# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

input_size = 224

binary = True
# last or best
last = True

output_base_dir = "runs\\train"
out_dir = os.path.join(output_base_dir,output_name)
if os.path.exists(out_dir):
  print("出力ディレクトリ名が被っているよ")
  sys.exit()

makedir(out_dir)

from choso_module import ai as chosoai

# グレースケール用のカスタマイズされたモデルを作成
model_ft = chosoai.ViT_l_16ForGrayscale(model_name='vit_base_patch16_224', pretrained=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_ft = model_ft.to(device)

