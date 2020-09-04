import argparse
import cv2
from model import Model
import time
import numpy as np
from create_data import get_data

def getArgument():
    arg = argparse.ArgumentParser()
    # định nghĩa một tham số cần parse
    arg.add_argument('-i', '--image_path',
                     help='link to image')
    # Giúp chúng ta convert các tham số nhận được thành một object và gán nó thành một thuộc tính của một namespace.
    return arg.parse_args()

# start time
start = time.time()

arg = getArgument()

# đọc được ảnh từ đường dẫn

img = cv2.imread(arg.image_path)

#hiển thị ảnh
cv2.imshow('Image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#resize ảnh về kích cỡ chuẩn 
resized = cv2.resize(img, (64,64), interpolation = cv2.INTER_AREA)


#dự đoán ảnh đó là ảnh gì
Model(False).predict(resized)

# end time
end = time.time()

# in ra thời gian thực hiện model
print('Model process on %.2f s' % (end - start))

#tinh toan evaluate cua mo hinh
print(Model(False).evaluate())
