import ai as chosoai
from torchvision import datasets, models, transforms
import csv
import numpy as np

def get_data_transforms(input_size, input_channels):
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
def main(data_dir, model_list, input_size, batch_size, epochs, output_name, input_channels=3, out_save = True):
    data_transforms = get_data_transforms(input_size, input_channels)
    
    with open('20230412-2.csv', 'w', newline="") as f: 
        writer = csv.writer(f)
        for model in model_list:
            cross_sum = 0
            csv_row_list = []
            for id,data in enumerate(data_dir):
                val_acc_hist = chosoai.train(data_dir=data,model_name=model,
                            output_name=  "resize_" + model+"-fold"+str(id+1),
                            num_classes=2,
                            batch_size = batch_size, num_epochs = epochs,
                            feature_extract = False, binary = (input_channels == 1), last = True, 
                            data_transforms = data_transforms
                            )

                csv_row_list.append(val_acc_hist)
        
            results_np = np.array(csv_row_list)
            mean_accuracies = results_np.mean(axis=0)
            best_epoch = np.argmax(mean_accuracies) + 1
            print(f"5つのfoldの精度の平均値が最も高いエポック: {best_epoch}")

            best_epoch_accuracies = results_np[:, best_epoch - 1]
            print(f"エポック {best_epoch} での5つのfoldの精度: {best_epoch_accuracies}")

            for i, acc in enumerate(best_epoch_accuracies, start=1):
                print(f"エポック {best_epoch} の fold {i} の精度: {acc}")
            print(f"平均値{np.mean(best_epoch_accuracies)}分散{np.var(best_epoch_accuracies)}")

 # epoch，feature_extract，binary，lastをここで設定する。
            #binaryはグレースケールか否か，グレースケールならTrue，カラーならFalse
            #lastがTrueなら最後のエポックのモデルを保存し，最後のエポックの混同行列を表示する。
            #lastがfalseなら最良のエポックとなる
if __name__ == '__main__':
    resize_data_dir = [r"D:\ex\egg\data\jpeg\resize\fold1", r"D:\ex\egg\data\jpeg\resize\fold2",
                r"D:\ex\egg\data\jpeg\resize\fold3", r"D:\ex\egg\data\jpeg\resize\fold4",
                r"D:\ex\egg\data\jpeg\resize\fold5"]
    model_list = ["efficientnetv2-s"]
    input_size = 384
    input_channels = 1  # 1 for grayscale, 3 for color
    batch_size = 16
    epochs = 30
    output_name = "resize" #アウトプット保存の名前
    main(resize_data_dir, model_list, input_size, batch_size, epochs, output_name, input_channels)

    halfway_data_dir = [r"D:\ex\egg\data\jpeg\halfway\fold1", r"D:\ex\egg\data\jpeg\halfway\fold2",
                r"D:\ex\egg\data\jpeg\halfway\fold3", r"D:\ex\egg\data\jpeg\halfway\fold4",
                r"D:\ex\egg\data\jpeg\halfway\fold5"]
    model_list = ["efficientnetv2-s"]
    input_size = 384
    output_name = "halfway" #アウトプット保存の名前
    main(halfway_data_dir, model_list, input_size, batch_size, epochs, output_name, input_channels)
    
    manual_data_dir = [r"D:\ex\egg\data\jpeg\manual\fold1", r"D:\ex\egg\data\jpeg\manual\fold2",
                r"D:\ex\egg\data\jpeg\manual\fold3", r"D:\ex\egg\data\jpeg\manual\fold4",
                r"D:\ex\egg\data\jpeg\manual\fold5"]
    model_list = ["efficientnetv2-s"]
    input_size = 384
    output_name = "manual" #アウトプット保存の名前
    main(manual_data_dir, model_list, input_size, batch_size, epochs, output_name, input_channels)

    model_list = ["efficientnetv2-m"]
    input_size = 480
    output_name = "resize" #アウトプット保存の名前
    main(resize_data_dir, model_list, input_size, batch_size, epochs, output_name, input_channels)

    model_list = ["efficientnetv2-m"]
    input_size = 480
    output_name = "halfway" #アウトプット保存の名前
    main(halfway_data_dir, model_list, input_size, batch_size, epochs, output_name, input_channels)
    
    model_list = ["efficientnetv2-m"]
    input_size = 480
    output_name = "manual" #アウトプット保存の名前
    main(manual_data_dir, model_list, input_size, batch_size, epochs, output_name, input_channels)

    model_list = ["efficientnet-b4"]
    input_size = 380
    output_name = "resize" #アウトプット保存の名前
    main(resize_data_dir, model_list, input_size, batch_size, epochs, output_name, input_channels)

    model_list = ["efficientnet-b4"]
    input_size = 380
    output_name = "halfway" #アウトプット保存の名前
    main(halfway_data_dir, model_list, input_size, batch_size, epochs, output_name, input_channels)
    
    model_list = ["efficientnet-b4"]
    input_size = 380
    output_name = "manual" #アウトプット保存の名前
    main(manual_data_dir, model_list, input_size, batch_size, epochs, output_name, input_channels)

    model_list = ["efficientnet-b5"]
    input_size = 456
    output_name = "resize" #アウトプット保存の名前
    main(resize_data_dir, model_list, input_size, batch_size, epochs, output_name, input_channels)

    model_list = ["efficientnet-b5"]
    input_size = 456
    output_name = "halfway" #アウトプット保存の名前
    main(halfway_data_dir, model_list, input_size, batch_size, epochs, output_name, input_channels)
    
    model_list = ["efficientnet-b5"]
    input_size = 456
    output_name = "manual" #アウトプット保存の名前
    main(manual_data_dir, model_list, input_size, batch_size, epochs, output_name, input_channels)

    model_list = ["efficientnet-b6"]
    input_size = 528
    output_name = "resize" #アウトプット保存の名前
    main(resize_data_dir, model_list, input_size, batch_size, epochs, output_name, input_channels)

    model_list = ["efficientnet-b6"]
    input_size = 528
    output_name = "halfway" #アウトプット保存の名前
    main(halfway_data_dir, model_list, input_size, batch_size, epochs, output_name, input_channels)
    
    model_list = ["efficientnet-b6"]
    input_size = 528
    output_name = "manual" #アウトプット保存の名前
    main(manual_data_dir, model_list, input_size, batch_size, epochs, output_name, input_channels)
    
    model_list = ["efficientnet-b3"]
    input_size = 300
    output_name = "resize" #アウトプット保存の名前
    main(resize_data_dir, model_list, input_size, batch_size, epochs, output_name, input_channels)

    model_list = ["efficientnet-b3"]
    input_size = 300
    output_name = "halfway" #アウトプット保存の名前
    main(halfway_data_dir, model_list, input_size, batch_size, epochs, output_name, input_channels)
    
    model_list = ["efficientnet-b3"]
    input_size = 300
    output_name = "manual" #アウトプット保存の名前
    main(manual_data_dir, model_list, input_size, batch_size, epochs, output_name, input_channels)

    model_list = ["efficientnet-b7"]
    input_size = 600
    output_name = "resize" #アウトプット保存の名前
    main(resize_data_dir, model_list, input_size, batch_size, epochs, output_name, input_channels)

    model_list = ["efficientnet-b7"]
    input_size = 600
    output_name = "halfway" #アウトプット保存の名前
    main(halfway_data_dir, model_list, input_size, batch_size, epochs, output_name, input_channels)
    
    model_list = ["efficientnet-b7"]
    input_size = 600
    output_name = "manual" #アウトプット保存の名前
    main(manual_data_dir, model_list, input_size, batch_size, epochs, output_name, input_channels)