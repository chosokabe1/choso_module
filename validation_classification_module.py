# from . import initialize_model
import initialize_model
# from . import ai as chosoai
import ai as chosoai
import torch 
import torch.nn as nn
import timm
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
import os


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
    
def validation(data_dir = "..", model_name = "aaa", output_name = "aaa", model_path = "aaa", num_classes = 2, batch_size = 2, num_epochs = 2, feature_extract = False, binary = True, last = True, data_transforms = {}, out_save=True, class_weights=None):
  if 'vit' in model_name:
    model_ft = ViT(model_name=model_name, binary=binary, pretrained=True, num_classes = num_classes)
    print(model_ft)
  else:
     model_ft, input_size = initialize_model.main(model_name, num_classes, feature_extract, use_pretrained=True, binary=binary)
  model_ft.load_state_dict(torch.load(model_path))

  # Create training and validation datasets
  image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['val']}
  # Create training and validation dataloaders
  dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['val']}
  class_names = image_datasets['val'].classes
  # Detect if we have a GPU available
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model_ft = model_ft.to(device)
  params_to_update = model_ft.parameters()
  optimizer_ft = torch.optim.Adam(params_to_update, lr=1e-4, weight_decay=1e-5)
  criterion = nn.CrossEntropyLoss()
  if out_save:
    output_base_dir = os.path.join("runs/test",output_name)
    output_directory = chosoai.create_output_directory(output_base_dir)
    os.makedirs(output_directory, exist_ok=True)
  confusion_matrix = chosoai.val_model(model_ft, dataloaders_dict, optimizer_ft, num_classes, criterion, binary, output_directory, class_names,out_save)

