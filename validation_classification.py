from torchvision import transforms
import validation_classification_module as choso_val

def get_data_transforms(input_size, input_channels):
    if input_channels == 1:
        data_transforms = {
            'val': transforms.Compose([
                transforms.Resize((input_size,input_size)),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]),
        }
    elif input_channels == 3:
        data_transforms = {
            'val': transforms.Compose([
                transforms.Resize((input_size,input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        }
    else:
        raise ValueError("Input channels should be either 1 or 3.")
    return data_transforms

def main(data_dir = "aaa", model_name= "aaa", input_size = 224, binary = False, model_path = "aaa", output_name = "aaa", input_channels=3, num_classes=1, out_save = True):
    data_transforms = get_data_transforms(input_size, input_channels)
    choso_val.validation(data_dir=data_dir, model_name=model_name,binary=binary,model_path=model_path,output_name=output_name,num_classes=num_classes,data_transforms=data_transforms)

if __name__ == '__main__':
    data_dir = r"D:\ex\shibuya\berry2023\note\ex5\test"
    model_name = "efficientnet-b0"
    input_size = 224
    model_path = r"D:\ex\shibuya\berry2023\note\runs\train\ex5_augmented_efficientnet-b0-fold1output_2\model_weights.pth"
    output_name = "ex5_augmentation_test"
    input_channels = 3
    num_classes = 5
    binary = False
    out_save = True

    main(data_dir=data_dir, model_name=model_name, binary=binary, input_size=input_size, model_path=model_path, output_name=output_name, input_channels=input_channels, num_classes=num_classes, out_save=out_save)