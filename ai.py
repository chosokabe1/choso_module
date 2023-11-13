from __future__ import print_function
from __future__ import division
from . import initialize_model
# import initialize_model
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import seaborn as sn
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
import timm


class ViT(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', binary = True, pretrained=True, num_classes=2):
        super(ViT, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        
        if binary:
          # グレースケール用にパッチ埋め込みレイヤーを変更
          self.model.patch_embed.proj = nn.Conv2d(1, self.model.patch_embed.proj.out_channels, kernel_size=self.model.patch_embed.proj.kernel_size, stride=self.model.patch_embed.proj.stride, padding=self.model.patch_embed.proj.padding)
        
        # 2クラス分類用に分類器を変更
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

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

def makedir(dir):
  if not os.path.exists(dir):
    os.makedirs(dir)

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
    print(device)
    since = time.time()

    train_acc_history = []
    val_acc_history = []
    model_wts = copy.deepcopy(model.state_dict())
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

            if phase == 'train':
                train_acc_history.append(epoch_acc)
            else:
                val_acc_history.append(epoch_acc)

            # deep copy the model
            # If last=true, update model_wts only when the last time, and if last=false, update the highest accuracy rate
            if phase == 'val':
                if last and epoch == num_epochs - 1:
                    model_wts = copy.deepcopy(model.state_dict())
                elif not last and epoch_acc > best_acc:
                    model_wts = copy.deepcopy(model.state_dict())
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(model_wts) # changing to a model instance
    return model, train_acc_history, val_acc_history

def tensor_to_np(inp, type):
    "imshow for Tesor"
    if type == "imagenet":
        inp = inp.numpy().transpose((1,2,0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        return inp

    elif type == "gray":
        inp = inp.numpy()
        inp = np.clip(inp, 0, 1)
        return inp

def false_img_save(pred, label, input, false_img_count, out_dir, class_names):
  pil_img = Image.fromarray(input)
  try:
    makedir(out_dir + 'error/pred_' + str(class_names[pred.item()]) + '_label_' + str(class_names[label.item()]))
    pil_img.save(out_dir + f'error/pred_{class_names[pred.item()]}_label_{class_names[label.item()]}/{false_img_count}.jpg')
  except IndexError:
    print(f'IndexError for pred {pred.item()} or label {label.item()}. Continuing...')

def save_txtfile(data, outpath):
  file = outpath
  fileobj = open(file, "w", encoding = "utf_8")
  for index , i in enumerate(data):
    if index == len(data) - 1:
      fileobj.write(f"{i}")
    else:
      fileobj.write(f"{i},")

  fileobj.close()

def val_model(model, dataloaders, optimizer, num_classes, criterion, binary, out_dir, class_names, out_save):
  false_img_count = 0
  confusion_matrix = torch.zeros(num_classes, num_classes)
  model.eval()
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  phase = 'val'
  for inputs, labels in dataloaders[phase]:
    inputs = inputs.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()
    with torch.set_grad_enabled(phase == 'train'):
      outputs = torch.nn.functional.softmax(model(inputs), dim=1)
      _, preds = torch.max(outputs, 1)
      for i in range(inputs.size()[0]):
        if preds[i] != labels[i]:
          if binary:
            type = "gray"
          else:
            type = "imagenet"
          input = tensor_to_np(inputs.cpu().data[i], type)
          input *= 255
          input = input.astype(np.uint8)
          if out_save:
            if type == "gray":
              input = np.squeeze(input)
            false_img_save(preds[i], labels[i], input, false_img_count, out_dir, class_names)
          false_img_count += 1

      for t_confusion_matrix, p_confusion_matrix in zip(labels.view(-1), preds.view(-1)):
        confusion_matrix[t_confusion_matrix.long(), p_confusion_matrix.long()] += 1
  confusion_matrix_numpy = confusion_matrix.to('cpu').detach().numpy().copy()
  if out_save:
    df_cmx = pd.DataFrame(confusion_matrix_numpy, index=class_names, columns=class_names)
    plt.figure(figsize = (12, 7))
    sn.set(font_scale = 1)
    sn.heatmap(df_cmx, annot=True, fmt='g', cmap='Blues')
    plt.savefig(os.path.join(out_dir,"confusion_matrix.png"))
    sn.set(font_scale = 1)
  return confusion_matrix_numpy

def create_output_directory(base_dir):
  i = 1
  while True:
    output_dir = base_dir + f'output_{i}'
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
      break
    i += 1
  return output_dir

def train(data_dir = "..", model_name = "aaa", output_name = "aaa", num_classes = 2, batch_size = 2, num_epochs = 2, feature_extract = False, binary = True, last = True, data_transforms = {}, out_save=True, class_weights=None):
  if 'vit' in model_name:
    model_ft = ViT(model_name=model_name, binary=binary, pretrained=True, num_classes = num_classes)
    print(model_ft)
  else:
     model_ft, input_size = initialize_model.main(model_name, num_classes, feature_extract, use_pretrained=True, binary=binary)
  
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model_ft = model_ft.to(device)

  # Create training and validation datasets
  image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
  # Create training and validation dataloaders
  dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
  class_names = image_datasets['train'].classes
  # Detect if we have a GPU available
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  params_to_update = model_ft.parameters()
  if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
      if param.requires_grad == True:
        params_to_update.append(param)
        # print("\t",name)
  # else:
    # for name,param in model_ft.named_parameters():
      # if param.requires_grad == True:
        # print("\t",name)

  # Observe that all parameters are being optimized
  # optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
  optimizer_ft = torch.optim.Adam(params_to_update, lr=1e-4, weight_decay=1e-5)
  # Setup the loss fxn
  if class_weights is None:
    criterion = nn.CrossEntropyLoss()
  else:
    criterion = nn.CrossEntropyLoss(weight=class_weights)

  # Train and evaluate
  model_ft, train_acc_hist_tensor, val_acc_hist_tensor = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, last, num_epochs=num_epochs, is_inception=(model_name=="inception"))
  if out_save:
    output_base_dir = os.path.join("runs/train",output_name)
    output_directory = create_output_directory(output_base_dir)
    makedir(output_directory)

  confusion_matrix = val_model(model_ft, dataloaders_dict, optimizer_ft, num_classes, criterion, binary, output_directory, class_names,out_save)

  train_acc_hist = []
  val_acc_hist = []

  for i in train_acc_hist_tensor:
    train_acc_hist.append(float(i.item()))

  for i in val_acc_hist_tensor:
    val_acc_hist.append(float(i.item()))

  if out_save:
    plt.figure(figsize = (12, 7))
    plt.plot(train_acc_hist, label="train_accuracy")
    plt.plot(val_acc_hist, label="test_accuracy")
    plt.title("Accuracy vs Epoch")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_directory, "acc.png"))
    save_txtfile(train_acc_hist, os.path.join(output_directory,"train_acc_hist.txt"))
    save_txtfile(val_acc_hist, os.path.join(output_directory,"val_acc_hist.txt"))
    torch.save(model_ft.state_dict(), os.path.join(output_directory,'model_weights.pth'))

  return val_acc_hist, confusion_matrix