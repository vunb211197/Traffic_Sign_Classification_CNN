# file này để tạo data
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
import config



def get_data():
    (X_train,y_train),(X_test,y_test)= mnist.load_data()
    #chia thành 2 tập train data và validation data
    X_val, y_val = X_train[40000:50000], y_train[40000:50000]
    X_train, y_train = X_train[:40000], y_train[:40000]

    y_train = np_utils.to_categorical(y_train, config.NUMBERS_CLASS)
    y_val = np_utils.to_categorical(y_val, config.NUMBERS_CLASS)
    y_test = np_utils.to_categorical(y_test, config.NUMBERS_CLASS)

    X_train=X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_val=X_val.reshape(X_val.shape[0], 28, 28, 1)
    X_test=X_test.reshape(X_test.shape[0], 28, 28, 1)
    return (X_train,y_train,X_val,y_val,X_test,y_test)