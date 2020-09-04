#file này để build,train , predict model 
import numpy as np
import matplotlib.pyplot as plt
import config
import cv2
from keras import optimizers
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten , BatchNormalization
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.models import Sequential
from create_data import get_data
from make_train_test_val_data import split_train_val_test_data

images=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22',
'23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42']

class Model():
    def __init__(self,trainable):
        # các thông số cho quá trình train
        self.batch_size = config.BATCH_SIZE
        self.trainable = False
        self.num_epochs = config.EPOCHS
        # Building model
        self.build_model()

        #kiểm tra nếu model chưa train thì bắt đầu train
        if trainable :
            self.model.summary()
            self.train()
        #nếu model đã train rồi thì load weights đã lưu ra để predict
        else :
            self.model.load_weights('weight.h5')

    def build_model(self):
        # CNN model
        '''' để có thể sử dụng API này trong keras thì input shape phải là 4 chiều '''
        self.model = Sequential()
        #thêm convolution layer
        self.model.add(Conv2D(16, (3, 3),padding='same', activation='relu', input_shape=(64,64,3)))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(16, (3, 3),padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        
        self.model.add(Conv2D(32, (3, 3),padding='same', activation='relu', input_shape=(64,64,3)))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, (3, 3),padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(64, (3, 3),padding='same', activation='relu', input_shape=(64,64,3)))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, (3, 3),padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))


        # FC
        self.model.add(Flatten())
        self.model.add(Dense(2048, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(config.NUMBERS_CLASS, activation='softmax'))

        self.model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(1e-4), metrics=['acc'])

    def train(self):
        # reduce learning rate
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, verbose=1,)
        # Model Checkpoint
        cpt_save = ModelCheckpoint('weight.h5', save_best_only=True, monitor='val_acc', mode='max')
        # lấy được dữ liệu 
        pixels,labels=get_data()
        X_train,y_train,X_val,y_val,X_test,y_test= split_train_val_test_data(pixels,labels)

        print("Training......")
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[cpt_save, reduce_lr], verbose=1,epochs=self.num_epochs, shuffle=True, batch_size=self.batch_size)

    def predict(self,img):
        y_predict = self.model.predict(img.reshape(1,64,64,3))
        print('Giá trị dự đoán: ', images[np.argmax(y_predict)])

    def evaluate(self):
        # lấy được dữ liệu 
        pixels,labels=get_data()
        X_train,y_train,X_val,y_val,X_test,y_test= split_train_val_test_data(pixels,labels)

        score = self.model.evaluate(X_test,y_test)
        return(score)
