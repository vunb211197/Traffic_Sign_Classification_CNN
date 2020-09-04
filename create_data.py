# file này đpc và xử lý d
import numpy as np
import config
import os
import pandas as pd
import cv2

data_path = 'GTSRB\\Final_Training\\Images'
input_size = (64,64)

def get_data():
    #khởi tạo các list để chứa labels và data
    pixels = []
    labels = []
    # lặp các thư mục trong data_path
    for dir in os.listdir(data_path):
        #lấy được tên các tệp trong thư mục
        class_dir = os.path.join(data_path, dir)
        #đọc file có đuôi csv để xử lý
        info_file = pd.read_csv(os.path.join(class_dir, 'GT-'+dir+'.csv'),sep=';')


        #lặp trong file csv 
        for row in info_file.iterrows():
            # Đọc ảnh
            pixel = cv2.imread(os.path.join(class_dir, row[1].Filename))
            # Trích phần ROI theo thông tin trong file csv
            pixel = pixel[row[1]['Roi.X1']:row[1]['Roi.X2'], row[1]['Roi.Y1']:row[1]['Roi.Y2'], :]

            # Resize về kích cỡ chuẩn
            img = cv2.resize(pixel, input_size)

            # Thêm vào list dữ liệu
            pixels.append(img)
 
            # Thêm nhãn cho ảnh
            labels.append(row[1].ClassId)
    #trả về pixels và labels
    return pixels,labels