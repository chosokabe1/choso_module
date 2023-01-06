from __future__ import print_function
from __future__ import division
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
from torchvision import datasets, models, transforms
import csv

def imgfullpaths2allcsv(imgfullpathlist, csv_path):
  with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["img_path"])
    for imgfullpath in imgfullpathlist:
      writer.writerow([os.path.basename(imgfullpath)])

def set_parameter_requires_grad(model, feature_extracting):
  if feature_extracting:
    for param in model.parameters():
      param.requires_grad = False

from efficientnet_pytorch import EfficientNet
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
  # Initialize these variables which will be set in this if statement. Each of these
  #   variables is model specific.
  model_ft = None
  input_size = 0
  if model_name == "resnet":
    """ Resnet18
    """
    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224

  elif model_name == "efficientnetb7":
    model_ft = EfficientNet.from_pretrained('efficientnet-b7')
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft._fc.in_features
    model_ft._fc = nn.Linear(num_ftrs, num_classes)
    input_size = 600

  elif model_name == "efficientnetv2m":
    model_ft = EfficientNet.from_pretrained('efficientnet-b7')
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft._fc.in_features
    model_ft._fc = nn.Linear(num_ftrs, num_classes)
    input_size = 480
    
  elif model_name == "alexnet":
    """ Alexnet
    """
    model_ft = models.alexnet(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
    input_size = 224

  elif model_name == "vgg":
    """ VGG11_bn
    """
    model_ft = models.vgg11_bn(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
    input_size = 224

  elif model_name == "squeezenet":
    """ Squeezenet
    """
    model_ft = models.squeezenet1_0(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    model_ft.num_classes = num_classes
    input_size = 224

  elif model_name == "densenet":
    """ Densenet
    """
    model_ft = models.densenet121(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)
    input_size = 224

  elif model_name == "inception":
    """ Inception v3
    Be careful, expects (299,299) sized images and has auxiliary output
    """
    model_ft = models.inception_v3(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    # Handle the auxilary net
    num_ftrs = model_ft.AuxLogits.fc.in_features
    model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
    # Handle the primary net
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs,num_classes)
    input_size = 299

  else:
    print("Invalid model name, exiting...")
    sys.exit()

  return model_ft, input_size

# データセットの定義
class only_img_Datasets(Dataset):
  def __init__(self, csv_file_all, data_path, data_transform, ):
    self.df_all = pd.read_csv(csv_file_all)
    self.data_path = data_path
    self.data_transform = data_transform

  def __len__(self):
    return len(self.df_all) # この長さを調節する必要がある(後で詳しく調べる)

  def __getitem__(self, i):
    pth = os.path.join(self.data_path, self.df_all['img_path'][i])
    tmp_img = Image.open(pth)
    tmp_img.convert("L").convert("RGB")
    inputs_img= self.data_transform(tmp_img)
    return inputs_img


def onlyio(model, dataloaders):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.eval()
  outputs_list = []
  for input in dataloaders:
    input = input.to(device)
    # optimizer.zero_grad()
    outputs = model(input)
    m = nn.Softmax(dim=1)
    outputs = m(outputs)
    outputs = outputs.to('cpu').detach().numpy().copy()
    outputs_list.append(outputs)

  return outputs_list