import ai as chosoai
from torchvision import datasets, models, transforms
import csv
import numpy as np

def main():
# 5分割交差検証　例えばfold1の下に各クラスのフォルダがある
    # data_dir = [r"D:\ex\egg\data\jpeg\halfway\fold1"
    #             ,r"D:\ex\egg\data\jpeg\halfway\fold2"
    #             ,r"D:\ex\egg\data\jpeg\halfway\fold3"
    #             ,r"D:\ex\egg\data\jpeg\halfway\fold4"
    #             ,r"D:\ex\egg\data\jpeg\halfway\fold5"]
    
    data_dir = [r"D:\ex\egg\data\dtd\train_val"]

    # model_list = ["efficientnet-b5"]
    # input_size = 456

    model_list = ["efficientnet-b0"]
    input_size = 224 # modelに応じて変える必要あり

    '''入力1channel'''
    # data_transforms = {
    #     'train': transforms.Compose([
    #         transforms.RandomHorizontalFlip(), # データ拡張
    #         transforms.RandomVerticalFlip(), # データ拡張
    #         transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # データ 拡張
    #         transforms.RandomRotation(degrees=(0,360),expand=True), # データ拡張
    #         transforms.Resize((input_size,input_size)),
    #         transforms.Grayscale(), # グレースケールの場合。
    #         transforms.ToTensor(), 
    #     ]),
    #     'val': transforms.Compose([
    #         transforms.Resize((input_size,input_size)),
    #         transforms.Grayscale(), # グレースケールの場合
    #         transforms.ToTensor(),
    #     ]),
    # }

    '''入力3channel'''
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((input_size,input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            # transforms.RandomRotation(degrees=(0,360),expand=True),
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

    with open('20230412-2.csv', 'w', newline="") as f: #　正解率推移をcsvに保存する
        writer = csv.writer(f)
        for model in model_list:
            cross_sum = 0
            csv_row_list = []
            for id,data in enumerate(data_dir):
                if "large" in model:
                    batch_size = 8
                else:
                    batch_size = 64 # バッチサイズはここで設定する

                val_acc_hist = chosoai.train(data_dir=data,model_name=model,
                            output_name=  "dtd_" + model+"-fold"+str(id+1),
                            num_classes=2,
                            batch_size = batch_size, num_epochs = 10,
                            feature_extract = False, binary = False, last = True, 
                            data_transforms = data_transforms
                            ) # epoch，feature_extract，binary，lastをここで設定する。
            #binaryはグレースケールか否か，グレースケールならTrue，カラーならFalse
            #lastがTrueなら最後のエポックのモデルを保存し，最後のエポックの混同行列を表示する。
            #lastがfalseなら最良のエポックとなる

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

if __name__ == '__main__':
    main()