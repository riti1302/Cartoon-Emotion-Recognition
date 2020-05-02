import os
import cv2
import pandas as pd

dataset_path = "training_dataset/"
df = pd.read_csv('../Face_detection/Train_annotated.csv')

for index, row in df.iterrows():
    os.makedirs(dataset_path + row['Emotion'], exist_ok = True)
    class_path = os.path.join(dataset_path, row['Emotion'])
    image = cv2.imread('../Data/Train/' + row['Frame_ID'])
    face = image[row['y_min']:row['y_max'], row['x_min']:row['x_max']]
    print(class_path + '/' + str(len(os.listdir(class_path))))
    cv2.imwrite(class_path + '/' + row['Frame_ID'], face)
