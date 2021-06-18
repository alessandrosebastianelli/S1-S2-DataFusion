from tensorflow.keras.layers import MaxPool2D, GlobalAveragePooling2D, BatchNormalization, Conv2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential 
from time import time
import numpy as np

class CNN_Classifier:
    def __init__(self, img_shape, n_classes):
        self.img_shape = img_shape
        self.n_classes = n_classes
        self.model = self.define_model()
        #self.model.compile(optimizer=Adam(lr=0.00001), loss='binary_crossentropy', metrics=['accuracy'])
        self.model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    def define_model(self):
        kernel_size = 5 
        stride_size = 1 
        pool_size = 2   

        model = Sequential()
        model.add(Conv2D(16, kernel_size, strides=stride_size, padding='same', activation='relu', input_shape=self.img_shape))
        model.add(MaxPool2D(pool_size=pool_size, strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size, strides=stride_size, activation='relu'))
        model.add(MaxPool2D(pool_size=pool_size, strides=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size, strides=stride_size, activation='relu'))
        model.add(MaxPool2D(pool_size=pool_size, strides=(2, 2)))
        model.add(BatchNormalization())
        #model.add(Conv2D(128, kernel_size, strides=stride_size, activation='relu'))
        #model.add(MaxPool2D(pool_size=pool_size, strides=(2, 2)))
        #model.add(BatchNormalization())
        model.add(GlobalAveragePooling2D())
        
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(self.n_classes, activation='sigmoid'))

        return model

    def train_model(self,  epochs, batch_size, train_data_generator, val_data_generator, training_steps, validation_steps):
        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
        
        history = self.model.fit(train_data_generator,
                                steps_per_epoch = training_steps, 
                                epochs=epochs,
                                validation_data=val_data_generator,
                                validation_steps=validation_steps,    
                                callbacks=[es],
                            )

        return history