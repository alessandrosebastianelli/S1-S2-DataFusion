from tensorflow.keras.layers import MaxPool2D, GlobalAveragePooling2D, BatchNormalization, Conv2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Concatenate, Add
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model 
from time import time
import numpy as np

class Fusion_Classifier:
    def __init__(self, img_a_shape, img_b_shape, n_classes):
        self.img_a_shape = img_a_shape
        self.img_b_shape = img_b_shape
        self.n_classes = n_classes
        self.model = self.define_model()
        #self.model.compile(optimizer=Adam(lr=0.00001), loss='binary_crossentropy', metrics=['accuracy'])
        self.model.compile(optimizer=RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    def define_model(self):
        kernel_size = 5 
        stride_size = 1 
        pool_size = 2   
        
        # Branch A
        xa_in = Input(shape=self.img_a_shape)

        xa = Conv2D(16, kernel_size, strides=stride_size, padding='same', activation='relu')(xa_in)
        xa = MaxPool2D(pool_size=pool_size, strides=(2, 2))(xa)
        xa = BatchNormalization()(xa)

        xa = Conv2D(32, kernel_size, strides=stride_size, padding='same', activation='relu')(xa)
        xa = MaxPool2D(pool_size=pool_size, strides=(2, 2))(xa)
        xa = BatchNormalization()(xa)

        xa = Conv2D(64, kernel_size, strides=stride_size, padding='same', activation='relu')(xa)
        xa = MaxPool2D(pool_size=pool_size, strides=(2, 2))(xa)
        xa = BatchNormalization()(xa)

        xa = GlobalAveragePooling2D()(xa)

        xa = Dense(64)(xa)
        xa = Dropout(0.3)(xa)

        # Branch B
        xb_in = Input(shape=self.img_b_shape)
        xb = Conv2D(16, kernel_size, strides=stride_size, padding='same', activation='relu')(xb_in)
        xb = MaxPool2D(pool_size=pool_size, strides=(2, 2))(xb)
        xb = BatchNormalization()(xb)

        xb = Conv2D(32, kernel_size, strides=stride_size, padding='same', activation='relu')(xb)
        xb = MaxPool2D(pool_size=pool_size, strides=(2, 2))(xb)
        xb = BatchNormalization()(xb)

        xb = Conv2D(64, kernel_size, strides=stride_size, padding='same', activation='relu')(xb)
        xb = MaxPool2D(pool_size=pool_size, strides=(2, 2))(xb)
        xb = BatchNormalization()(xb)

        xb = GlobalAveragePooling2D()(xb)

        xb = Dense(64)(xb)
        xb = Dropout(0.3)(xb)

        # Concatenate
        x = Concatenate()([xb, xa])
        x = Dropout(0.3)(x)
        #x = Dense(self.n_classes*4, activation='relu')(x)
        #x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.3)(x)

        x = Dense(16, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(self.n_classes, activation='sigmoid')(x)

        model = Model(inputs = [xa_in, xb_in], outputs = x)

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