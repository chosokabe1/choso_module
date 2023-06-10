from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
from choso_module import ai as chosoai
import cv2
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from PIL import Image
import numpy as np
import os
# Register hook
feature_map = None
def hook(module, input, output):
    global feature_map
    feature_map = output.detach()

def visualize_feature_map_img(model, layer_name, binary, input_size, data_transforms, image_path, save_dir_path):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(DEVICE)

    # Access the layer of interest
    if layer_name == '_conv_stem':
        layer = model._conv_stem
    elif layer_name == '_conv_head':
        layer = model._conv_head
    elif layer_name.startswith('_blocks'):
        # _blocksの指定は_blocks[index]という形で入力する
        index = int(layer_name.split("[")[1].split("]")[0])
        layer = model._blocks[index]
    else:
        print(f"Invalid layer name: {layer_name}")
        return
    
    if binary:
        image = cv2.imread(image_path,0)
        input_image = Image.fromarray(image)

    else:
        image = cv2.imread(image_path,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (input_size, input_size)) 
        input_image = Image.fromarray(image)
        vis_image = image / 255.0 #(height, width, channel), [0, 1]

    input_image = data_transforms(input_image).unsqueeze(0).to(DEVICE)
    handle = layer.register_forward_hook(hook)
    # Forward pass
    out = model(input_image)

    # Remove the hook
    handle.remove()

    # Calculate the number of rows and columns needed
    num_filters = feature_map.shape[1]
    num_cols = int(np.sqrt(num_filters))
    num_rows = num_filters // num_cols + int(num_filters % num_cols > 0)

    # Plotting the feature map
    plt.figure(figsize=(20, 20))
    for i, filter in enumerate(feature_map.squeeze(0)):
        plt.subplot(num_rows, num_cols, i+1)
        filter = filter - filter.min()
        filter = filter / filter.max()
        plt.imshow(filter.cpu().numpy(), cmap='gray')
        plt.axis("off")


    filename = os.path.basename(image_path).split(".")[0] # getting filename without extension
    plt.savefig(os.path.join(save_dir_path, f'{filename}_feature_map.png')) # save the figure to file
    plt.close() # close the figure

def visualize_feature_map_dir(model_name, model_path, layer_name, num_classes, binary, image_dir_path, save_dir_path):
    os.makedirs(save_dir_path, exist_ok=True)
    if binary == True:
        model, input_size = chosoai.initialize_model(model_name=model_name, num_classes=num_classes, feature_extract=False, use_pretrained=True, binary=True)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        data_transforms = transforms.Compose([
            transforms.Resize(input_size),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

    else:
        model, input_size = chosoai.initialize_model(model_name=model_name, num_classes=num_classes, feature_extract=False, use_pretrained=True)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        data_transforms = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Iterate over all files in the image directory
    for filename in os.listdir(image_dir_path):
        # Check if the file is an image
        if filename.endswith('.jpeg') or filename.endswith('.jpg') or filename.endswith('.png'):
            # Create a full path to the image file
            image_path = os.path.join(image_dir_path, filename)
            # Apply visualize feature map and save the result
            visualize_feature_map_img(model, layer_name, binary, input_size, data_transforms, image_path, save_dir_path)