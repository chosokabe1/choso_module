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
import time
import copy

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
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True, binary=False):
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
    if binary:
      model_ft.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=model_ft.conv1.kernel_size,
        stride=model_ft.conv1.stride,
        padding=model_ft.conv1.padding,
        bias=False
      )
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

def classio(model, dataloaders):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.eval()
  preds_list = []
  for input in dataloaders:
    input = input.to(device)
    # optimizer.zero_grad()
    outputs = model(input)
    m = nn.Softmax(dim=1)
    outputs = m(outputs)
    _, preds = torch.max(outputs, 1)

    preds = preds.to('cpu').detach().numpy().copy()
    preds_list.append(preds)

  return preds_list
def henkou_num_class(model, num_class, feature_extract):
  set_parameter_requires_grad(model, feature_extract)
  num_ftrs = model.fc.in_features
  model.fc = nn.Linear(num_ftrs, num_class)

  return model

def train_model(model, dataloaders, criterion, optimizer, last, num_epochs=25, is_inception=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    since = time.time()

    train_acc_history = []
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    last_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            #######
            # 追加
            #######
            # if phase == 'val':
            #     print('{} / {} = Acc: {:.4f}'.format(running_corrects.double(), len(dataloaders[phase].dataset), epoch_acc))

            #     print(preds)

            #     print(labels.data)

            if phase == 'train':
                train_acc_history.append(epoch_acc)
            #######

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            
            last_model_wts = copy.deepcopy(model.state_dict())

        print()

    last_model_wts = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    if last:
        model.load_state_dict(last_model_wts)
    else:
        model.load_state_dict(best_model_wts)

    return model, train_acc_history, val_acc_history