from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
import ai as chosoai
import cv2
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
from PIL import Image
import numpy as np
import os

def classification_img(model, binary, input_size, data_transforms, image_path, save_file_path):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(DEVICE)

    if binary:
        image = cv2.imread(image_path,0)
        input_image = Image.fromarray(image)

    else:
        image = cv2.imread(image_path,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (input_size, input_size)) 
        input_image = Image.fromarray(image)

    input_image = data_transforms(input_image).unsqueeze(0).to(DEVICE)

    # Forward Pass to get the prediction
    with torch.no_grad():
        model.eval()
        output = model(input_image)
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
            classification_img(model, binary, input_size, data_transforms, image_path, save_file_path)