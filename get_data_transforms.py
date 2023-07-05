from torchvision import datasets, models, transforms

def main(input_size, input_channels):
    if input_channels == 1:
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
    elif input_channels == 3:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((input_size,input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
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
    else:
        raise ValueError("Input channels should be either 1 or 3.")
    return data_transforms