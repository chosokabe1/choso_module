from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
# import ai as chosoai
import initialize_model
import cv2
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from PIL import Image
import numpy as np
import os
import sys

def classification_img(model, binary, input_size, data_transforms, image_path, save_file_path):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(DEVICE)
    if binary:
        input_image = Image.open(image_path).convert('l')
    else:
        input_image = Image.open(image_path).convert('RGB')
    input_image = data_transforms['val'](input_image).unsqueeze(0).to(DEVICE)

    # Forward Pass to get the prediction
    with torch.no_grad():
        model.eval()
        output1 = model(input_image)
        output = torch.nn.functional.softmax(output1, dim=1)
        _, pred = torch.max(output, 1)
    
    # Get the basename of the image file and the predicted class
    basename = os.path.basename(image_path)
    predicted_class = pred.item()

    # Append this data to the CSV file
    with open(save_file_path, 'a') as f:
        f.write(f'{basename},{predicted_class}\n')

def classification_dir(model_name, model_path, num_classes, binary, image_dir_path, save_file_path):
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
    if binary == True:
        model, input_size = initialize_model.main(model_name=model_name, num_classes=num_classes, feature_extract=False, use_pretrained=True, binary=True)
    else:
        model, input_size = initialize_model.main(model_name=model_name, num_classes=num_classes, feature_extract=False, use_pretrained=True)
        
    model.load_state_dict(torch.load(model_path))
    if binary:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((input_size,input_size)),
                transforms.RandomHorizontalFlip(), 
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.RandomRotation(degrees=(0,360),expand=True),
                transforms.Resize((input_size,input_size)),
                transforms.Grayscale(),
                transforms.ToTensor(), 
            ]),
            'val': transforms.Compose([
                transforms.Resize((input_size,input_size)),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]),
        }
    else:
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomVerticalFlip(),
                # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.Resize((input_size,input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((input_size,input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        }


    # Iterate over all files in the image directory
    for filename in os.listdir(image_dir_path):
        # Check if the file is an image
        if filename.endswith('.jpeg') or filename.endswith('.jpg') or filename.endswith('.png'):
            # Create a full path to the image file
            image_path = os.path.join(image_dir_path, filename)
            # Apply visualize feature map and save the result
            classification_img(model, binary, input_size, data_transforms, image_path, save_file_path)

def str2bool(s):
    return s.lower() in ["true"]

if __name__ == "__main__":
    classification_dir(sys.argv[1],sys.argv[2],int(sys.argv[3]),str2bool(sys.argv[4]),sys.argv[5],sys.argv[6])