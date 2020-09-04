from keras.utils import np_utils
import numpy as np
#class để phân chia thành các tập val , test, train từ một bộ dữ liệu lớn

def split_train_val_test_data(pixels, labels):
    # Chuẩn hoá dữ liệu pixels và labels

    #đưa dữ liệu về mảng np array
    pixels = np.array(pixels)

    #đưa nhãn về dạng one-hot vector
    labels = np_utils.to_categorical(labels)

    # Nhào trộn dữ liệu ngẫu nhiên để cho công bằng
    randomize = np.arange(len(pixels))
    np.random.shuffle(randomize)

    X = pixels[randomize]
    y = labels[randomize]

    # Chia dữ liệu theo tỷ lệ 60% train và 40% còn lại cho val và test
    train_size = int(X.shape[0] * 0.6)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    val_size = int(X_val.shape[0] * 0.5) # 50% của phần 40% bên trên
    X_val, X_test = X_val[:val_size], X_val[val_size:]
    y_val, y_test = y_val[:val_size], y_val[val_size:]

    #trả về các tập thu được 
    return X_train, y_train, X_val, y_val, X_test, y_test
