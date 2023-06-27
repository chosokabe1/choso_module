import ai as chosoai
from torchvision import datasets, models, transforms
import csv
import numpy as np

# 5分割交差検証　例えばfold1の下に各クラスのフォルダがある
data_dir = ["D:\\ex\\data\\rectangle\\fold1"
            ,"D:\\ex\\data\\rectangle\\fold2"
            ,"D:\\ex\\data\\rectangle\\fold3"
            ,"D:\\ex\\data\\rectangle\\fold4"
            ,"D:\\ex\\data\\rectangle\\fold5"]

model_list = ["efficientnet-b5"]
input_size = 456
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomRotation(degrees=(0,360),expand=True),
        transforms.Resize(input_size),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ]),
}
with open('20230412-2.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    for model in model_list:
        cross_sum = 0
        csv_row_list = []
        for id,data in enumerate(data_dir):
            if "large" in model:
                batch_size = 8
            else:
                batch_size = 8

            val_acc_hist = chosoai.train(data_dir=data,model_name=model,
                          output_name=  "maxsquare_0412_" + model+"-fold"+str(id+1),
                          num_classes=2,
                          batch_size = batch_size, num_epochs = 50,
                          feature_extract = False, input_size = 224, 
                          binary = True, last = True, 
                          data_transforms = data_transforms
                        )

            csv_row_list.append(val_acc_hist)
        
        results_np = np.array(csv_row_list)
        # 各エポックにおける5つのfoldの精度の平均値を計算
        mean_accuracies = results_np.mean(axis=0)
        # 平均値が最大となるエポックを見つける
        best_epoch = np.argmax(mean_accuracies) + 1
        print(f"5つのfoldの精度の平均値が最も高いエポック: {best_epoch}")

        # そのエポックでの5つのfoldの精度を取得し、表示
        best_epoch_accuracies = results_np[:, best_epoch - 1]
        print(f"エポック {best_epoch} での5つのfoldの精度: {best_epoch_accuracies}")

        for i, acc in enumerate(best_epoch_accuracies, start=1):
            print(f"エポック {best_epoch} の fold {i} の精度: {acc}")
        print(f"平均値{np.mean(best_epoch_accuracies)}分散{np.var(best_epoch_accuracies)}")