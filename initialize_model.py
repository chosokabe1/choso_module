from efficientnet_pytorch import EfficientNet
from torchvision import datasets, models, transforms
import torch.nn as nn
import sys
import timm
import torch
from typing import Tuple, Optional
import torch.nn.functional as F

def _calc_same_pad(i: int, k: int, s: int, d: int):
    return max((-(i // -s) - 1) * s + (k - 1) * d + 1 - i, 0)

def conv2d_same(
        x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0), dilation: Tuple[int, int] = (1, 1), groups: int = 1):
    ih, iw = x.size()[-2:]
    kh, kw = weight.size()[-2:]
    pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
    pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
    x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)

class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    # pylint: disable=unused-argument
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
def set_parameter_requires_grad(model, feature_extracting):
  if feature_extracting:
    for param in model.parameters():
      param.requires_grad = False

def main(model_name, num_classes, feature_extract, use_pretrained=True, binary=False):
  # Initialize these variables which will be set in this if statement. Each of these
  #   variables is model specific.
  model_ft = None
  input_size = 0
  if "resnet" in model_name:
    if model_name == "resnet18":
      model_ft = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
      num_ftrs = model_ft.fc.in_features
    
    elif model_name == "resnet34":
      model_ft = models.resnet34(pretrained=use_pretrained)
      num_ftrs = model_ft.fc.in_features

    elif model_name == "resnet50":
      model_ft = models.resnet50(pretrained=use_pretrained)
      num_ftrs = model_ft.fc.in_features

    set_parameter_requires_grad(model_ft, feature_extract)
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

      # elif "swin" in model_name:
  #   if model_name == "swin_t":
  #       model_ft = SwinTransformer(hidden_dim=96,
  #                                  layers=(2, 2, 6, 2),
  #                                  heads=(3, 6, 12, 24),
  #                                  channels=3,
  #                                  num_classes=2,
  #                                  head_dim=32,
  #                                  window_size=7,
  #                                  downscaling_factors=(4, 2, 2, 2),
  #                                  relative_pos_embedding=True)
        
  #       input_size = 224

  #   elif model_name == "swin_v2_t":
  #       model_ft = SwinTransformer(img_size=224, patch_size=4, in_chans=3,
  #                                  num_classes=num_classes, embed_dim=96,
  #                                  depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
  #                                  window_size=7, mlp_ratio=4, qkv_bias=True,
  #                                  use_absolute_pos_emb=True, drop_path_rate=0.1)
  #       input_size = 224

  
    if binary:
        model_ft.patch_embed = nn.Conv2d(1, model_ft.embed_dim, kernel_size=model_ft.patch_size, stride=model_ft.patch_size, padding=0)

    set_parameter_requires_grad(model_ft, feature_extract)

  elif model_name == "vit_l_16":
    model_ft = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
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
    input_size = 480

  elif "efficientnet-b" in model_name:
    first_layer_out_channels_map = {
        "efficientnet-b0": 32,
        "efficientnet-b1": 32,
        "efficientnet-b2": 32,
        "efficientnet-b3": 40,
        "efficientnet-b4": 48,
        "efficientnet-b5": 48,
        "efficientnet-b6": 56,
        "efficientnet-b7": 64,
    }
    if model_name == "efficientnet-b0":
      model_ft = EfficientNet.from_pretrained('efficientnet-b0')
      input_size = 224

    elif model_name == "efficientnet-b1":
      model_ft = EfficientNet.from_pretrained('efficientnet-b1')
      input_size = 240

    elif model_name == "efficientnet-b2":
      model_ft = EfficientNet.from_pretrained('efficientnet-b2')
      input_size = 260
  
    elif model_name == "efficientnet-b3":
      model_ft = EfficientNet.from_pretrained('efficientnet-b3')
      input_size = 300
  
    elif model_name == "efficientnet-b4":
      model_ft = EfficientNet.from_pretrained('efficientnet-b4')
      input_size = 380

    elif model_name == "efficientnet-b5":
      model_ft = EfficientNet.from_pretrained('efficientnet-b5')
      input_size = 456

    elif model_name == "efficientnet-b6":
      model_ft = EfficientNet.from_pretrained('efficientnet-b6')
      input_size = 528

    elif model_name == "efficientnet-b7":
      model_ft = EfficientNet.from_pretrained('efficientnet-b7')
      input_size = 600

    if binary:
      out_channels = first_layer_out_channels_map[model_name]
      model_ft._conv_stem = nn.Conv2d(1, out_channels, kernel_size=3, stride=2, bias=False)

    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft._fc.in_features
    model_ft._fc = nn.Linear(num_ftrs, num_classes)

  elif "efficientnetv" in model_name:
    first_layer_out_channels_map = {
      "efficientnetv2-s": 24,
      "efficientnetv2-m": 24,
      "efficientnetv2-l": 32,
    }
    if model_name == "efficientnetv2-s":
      model_ft = timm.create_model('tf_efficientnetv2_s', pretrained=True)
      input_size = 384  # EfficientNetV2-Sのデフォルト入力サイズ
    elif model_name == "efficientnetv2-m":
      model_ft = timm.create_model('tf_efficientnetv2_m', pretrained=True)
      input_size = 480  # EfficientNetV2-mのデフォルト入力サイズ
    elif model_name == "efficientnetv2-l":
      model_ft = timm.create_model('tf_efficientnetv2_l', pretrained=True)
      input_size = 480  # EfficientNetV2-lのデフォルト入力サイズ

    if binary:
      out_channels = first_layer_out_channels_map[model_name]
      model_ft.conv_stem = Conv2dSame(1, out_channels, kernel_size=3, stride=2, bias=False)

    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier.in_features
    model_ft._fc = nn.Linear(num_ftrs, num_classes)
    
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