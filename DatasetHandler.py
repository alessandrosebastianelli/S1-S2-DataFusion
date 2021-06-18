from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import rasterio
import random
import glob
import cv2
import os

class DatasetHandler:
    def __init__(self, root):
        self.s2_paths, self.s2_labels, self.classes = self.__load_paths_labels(root, 'sentinel2')
        self.s1_paths, self.s1_labels, _ = self.__load_paths_labels(root, 'sentinel1')

        # Shuffle paths and labels in the same way
        c = list(zip(self.s2_paths, self.s2_labels, self.s1_paths, self.s1_labels))
        random.shuffle(c)
        self.s2_paths, self.s2_labels, self.s1_paths, self.s1_labels = zip(*c)

        #self.s2_shape = self.__load_data(self.s2_paths[0]).shape
        #self.s1_shape = self.__load_data(self.s1_paths[0]).shape

    def __load_paths_labels(self, root, platform):
        root2 = os.path.join(os.path.join(root, platform), '*')
        classes = glob.glob(root2)
        classes.sort()

        paths = []
        labels = []
        for i, cl in enumerate(classes):
            
            parths_in_cl = glob.glob(os.path.join(cl, '*'))
            parths_in_cl.sort()

            for img in parths_in_cl:
                paths.append(img)
                l = np.zeros((len(classes)))
                l[i] = 1
                labels.append(l)
        
        paths = np.array(paths, dtype=object)
        labels = np.array(labels)

        return paths, labels, classes

    def __load_data(self, path):
        with rasterio.open(path) as src:
            data = src.read()
        data = np.transpose(data)

        return data

    def __normalize(self, data):
        for i in range(data.shape[-1]):
            data[...,i] = (data[...,i] - np.min(data[...,i]))/((np.max(data[...,i])-np.min(data[...,i]))+1e-6)
            infs = (data[...,i] == np.inf) 
            nans = np.isnan(data[...,i])
            
            if infs.any() or nans.any():
                data[...,i] = np.ones((data[...,i].shape[0], data[...,i].shape[1]))

        return data 

    def s2_data_loader(self, batch_size, shape):
        augmentator = ImageDataGenerator(
            rotation_range=45,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=[0.3, 1.],
            horizontal_flip=True,
            vertical_flip=True,
            #brightness_range=[0.8, 1.2],
            fill_mode='reflect',
        )

        b_in = np.zeros((batch_size, shape[0], shape[1], shape[2]))
        b_out = np.zeros((batch_size, len(self.classes)))

        indexes = random.sample(range(len(self.s2_paths)), len(self.s2_paths)-1)
        counter = 0

        while True:
            for i in range(batch_size):

                if counter >= len(self.s2_paths) - 1:
                    indexes = random.sample(range(len(self.s2_paths)), len(self.s2_paths)-1)
                    counter = 0

                #print(counter, 'of', len(self.s1_paths), indexes[counter])

                data = self.__load_data(self.s2_paths[indexes[counter]])[...,:12]
                data = self.__normalize(data)
                
                data = cv2.resize(data, dsize=(shape[0], shape[1]), interpolation=cv2.INTER_CUBIC)

                b_in[i, ...] = 2.5*data#[:shape[0], :shape[1], ...]
                # Data augmentation
                b_in[i, ...] = augmentator.apply_transform(b_in[i, ...], augmentator.get_random_transform(shape))
                b_in[i, ...] = np.clip(b_in[i, ...], 0.0, 1.0)
                
                b_out[i, ...] = self.s2_labels[indexes[counter]]

                counter += 1


            yield b_in, b_out

    def s1_data_loader(self, batch_size, shape):
        augmentator = ImageDataGenerator(
            rotation_range=45,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=[0.3, 1.],
            horizontal_flip=True,
            vertical_flip=True,
            #brightness_range=[0.8, 1.2],
            fill_mode='reflect',
        )

        b_in = np.zeros((batch_size, shape[0], shape[1], shape[2]))
        b_out = np.zeros((batch_size, len(self.classes)))

        indexes = random.sample(range(len(self.s1_paths)), len(self.s1_paths)-1)
        counter = 0

        while True:
            for i in range(batch_size):

                if counter >= len(self.s1_paths) - 1:
                    indexes = random.sample(range(len(self.s1_paths)), len(self.s1_paths)-1)
                    counter = 0

                data = self.__load_data(self.s1_paths[indexes[counter]])

                data = np.clip(data, -30.0, 15.0)
                data = self.__normalize(data)
                #print(self.s1_paths[indexes[counter]])
                data = cv2.resize(data, dsize=(shape[0], shape[1]), interpolation=cv2.INTER_CUBIC)

                b_in[i, ...] = data#[:shape[0], :shape[1], ...]
                # Data augmentation
                b_in[i, ...] = augmentator.apply_transform(b_in[i, ...], augmentator.get_random_transform(shape))
                b_in[i, ...] = np.clip(b_in[i, ...], 0.0, 1.0)
                
                b_out[i, ...] = self.s1_labels[indexes[counter]]

                counter += 1

            yield b_in, b_out
        
    def s2_s1_data_loader(self, batch_size, shape_a, shape_b):
        augmentator = ImageDataGenerator(
            rotation_range=45,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=[0.3, 1.],
            horizontal_flip=True,
            vertical_flip=True,
            #brightness_range=[0.8, 1.2],
            fill_mode='reflect',
        )

        b_in_a = np.zeros((batch_size, shape_a[0], shape_a[1], shape_a[2]))
        b_in_b = np.zeros((batch_size, shape_b[0], shape_b[1], shape_b[2]))
        b_out = np.zeros((batch_size, len(self.classes)))

        indexes = random.sample(range(len(self.s1_paths)), len(self.s1_paths)-1)
        counter = 0

        while True:
            for i in range(batch_size):

                if counter >= len(self.s1_paths) - 1:
                    indexes = random.sample(range(len(self.s1_paths)), len(self.s1_paths)-1)
                    counter = 0

                transform = augmentator.get_random_transform((shape_a[0], shape_a[1]))

                # Load Sentinel-1
                data = self.__load_data(self.s1_paths[indexes[counter]])
                data = np.clip(data, -30.0, 10.0)
                data = self.__normalize(data)
                data = cv2.resize(data, dsize=(shape_b[0], shape_a[1]), interpolation=cv2.INTER_CUBIC)
                b_in_b[i, ...] = data#[:shape[0], :shape[1], ...]
                # Data augmentation
                b_in_b[i, ...] = augmentator.apply_transform(b_in_b[i, ...], transform)
                b_in_b[i, ...] = np.clip(b_in_b[i, ...], 0.0, 1.0)
                
                # Load Sentinel-2
                data = self.__load_data(self.s2_paths[indexes[counter]])[...,:12]
                data = self.__normalize(data)
                
                data = cv2.resize(data, dsize=(shape_a[0], shape_a[1]), interpolation=cv2.INTER_CUBIC)

                b_in_a[i, ...] = 2.5*data#[:shape[0], :shape[1], ...]
                # Data augmentation
                b_in_a[i, ...] = augmentator.apply_transform(b_in_a[i, ...], transform)
                b_in_a[i, ...] = np.clip(b_in_a[i, ...], 0.0, 1.0)
                
    
                
                b_out[i, ...] = self.s1_labels[indexes[counter]]

                counter += 1
            

            yield [b_in_a, b_in_b], b_out

    def s2_s1_data_loader_2(self, batch_size, shape_a, shape_b):
        augmentator = ImageDataGenerator(
            rotation_range=45,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=[0.3, 1.],
            horizontal_flip=True,
            vertical_flip=True,
            #brightness_range=[0.8, 1.2],
            fill_mode='reflect',
        )

        b_in = np.zeros((batch_size, shape_a[0], shape_a[1], shape_a[2]+shape_b[2]))
        b_out = np.zeros((batch_size, len(self.classes)))

        indexes = random.sample(range(len(self.s1_paths)), len(self.s1_paths)-1)
        counter = 0

        while True:
            for i in range(batch_size):

                if counter >= len(self.s1_paths) - 1:
                    indexes = random.sample(range(len(self.s1_paths)), len(self.s1_paths)-1)
                    counter = 0

                transform = augmentator.get_random_transform((shape_a[0], shape_a[1]))

                # Load Sentinel-1
                data = self.__load_data(self.s1_paths[indexes[counter]])
                data = np.clip(data, -30.0, 10.0)
                data = self.__normalize(data)
                data = cv2.resize(data, dsize=(shape_b[0], shape_a[1]), interpolation=cv2.INTER_CUBIC)
                b_in[i, ..., 0:shape_b[2]] = data#[:shape[0], :shape[1], ...]
                # Data augmentation
                b_in[i, ...,0:shape_b[2]] = augmentator.apply_transform(b_in[i, ...,0:shape_b[2]], transform)
                b_in[i, ...,0:shape_b[2]] = np.clip(b_in[i, ...,0:shape_b[2]], 0.0, 1.0)
                
                # Load Sentinel-2
                data = self.__load_data(self.s2_paths[indexes[counter]])[...,:12]
                data = self.__normalize(data)
                
                data = cv2.resize(data, dsize=(shape_a[0], shape_a[1]), interpolation=cv2.INTER_CUBIC)

                b_in[i, ...,shape_b[2]:] = 2.5*data#[:shape[0], :shape[1], ...]
                # Data augmentation
                b_in[i, ..., shape_b[2]:] = augmentator.apply_transform(b_in[i, ..., shape_b[2]:], transform)
                b_in[i, ..., shape_b[2]:] = np.clip(b_in[i, ..., shape_b[2]:], 0.0, 1.0)
                
                b_out[i, ...] = self.s1_labels[indexes[counter]]

                counter += 1
            

            yield b_in, b_out