import ai as chosoai
from torchvision import datasets, models, transforms
import csv
import numpy as np
import get_data_transforms
import csv
import os

def create_output_csv_name(base_name, max_attempts=1000):
    for i in range(1, max_attempts+1):
        output_file = base_name + f'_output_{i}.csv'
        if not os.path.exists(output_file):
            return output_file
    raise ValueError(f"Failed to create CSV filename after {max_attempts} attempts.")

def main(data_dir, model_name, input_size, batch_size, epochs, output_name, input_channels=3, repeat_number=10):
    data_transforms = get_data_transforms.main(input_size, input_channels)
    out_csv_base_name = output_name + '_' + model_name + '_iter_' + str(repeat_number)
    out_csv_name = create_output_csv_name(out_csv_base_name)
    header = []
    for i in range(1, len(data_dir)+1):
        for j in range(2):  # Assuming binary classification (num_classes=2)
            header.append(f'dataset{i}_class{j}')
    
    for repeat in range(repeat_number):
        with open(out_csv_name, 'a', newline="") as f: 
            writer = csv.writer(f)
            if repeat == 0:
                writer.writerow(header)  # Write the header in the first repeat

            row = []
            for id, data in enumerate(data_dir):
                val_acc_hist, confusion_matrix = chosoai.train(
                    data_dir=data,
                    model_name=model_name,
                    output_name=output_name + model_name + "-fold" + str(id + 1),
                    num_classes=2,
                    batch_size=batch_size,
                    num_epochs=epochs,
                    feature_extract=False,
                    binary=(input_channels == 1),
                    last=True,
                    data_transforms=data_transforms,
                    out_save=False
                )

                # Assuming that the classes are labeled 0 and 1
                row.append(confusion_matrix[0, 0])  # True positives for class 0
                row.append(confusion_matrix[1, 1])  # True positives for class 1
                print(confusion_matrix)

            writer.writerow(row)

if __name__ == '__main__':
    resize_data_dir = [r"D:\ex\egg\data\jpeg\rectangle\fold1", r"D:\ex\egg\data\jpeg\rectangle\fold2",
                r"D:\ex\egg\data\jpeg\rectangle\fold3", r"D:\ex\egg\data\jpeg\rectangle\fold4",
                r"D:\ex\egg\data\jpeg\rectangle\fold5"]
    # model_name = "efficientnetv2-l"
    model_name = "efficientnet-b0"
    input_size = 224
    input_size = 480
    input_channels = 1  # 1 for grayscale, 3 for color
    batch_size = 16
    epochs = 3
    output_name = "resize" #アウトプット保存の名前
    main(resize_data_dir, model_name, input_size, batch_size, epochs, output_name, input_channels, repeat_number=3)