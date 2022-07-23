from tensorflow.keras.preprocessing.image import ImageDataGenerator
from numpy.random import randint
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import rasterio
import random
import os

def get_images_path(path, satellite):
    '''
        It returns a list of path.
        Satellite must be 'sen1' or 'sen2'.
        Path must be the master folder with 'sen1' and 'sen2' folder inside
    '''
    path = os.path.join(path, satellite)

    images_path = []

    zones = os.listdir(path)
    for zone in zones:
        if not '.DS_' in zone:
            zone_path = os.path.join(path, zone)
            images = os.listdir(zone_path)

        for image in images:
            if not '.DS_' in image:
                image_path = os.path.join(zone_path, image)
                images_path.append(image_path)

    images_path.sort()
    return images_path, zones

def get_s2_image(image_path, normalization='minmax'):
    '''
        It returns an RGB image using a path.
        Image path must be the full path of an RGB image (tif format)
    '''

    # rasterio function to read bands of a tif file
    dataset = rasterio.open(image_path)
    r = dataset.read(1)
    g = dataset.read(2)
    b = dataset.read(3)
    dataset.close()

    # RGB composite
    rgb = np.zeros((r.shape[0],r.shape[1], 3))
    rgb[...,0] = r
    rgb[...,1] = g
    rgb[...,2] = b

    if normalization == 'minmax':
      # Normalization
      rgb = (rgb - rgb.min())/(rgb.max() - rgb.min())
    elif normalization =='std':
      # Standardization
      rgb = (rgb - rgb.mean())/(rgb.std())

    rgb = np.clip(rgb, 0.0, 1.0)

    return rgb

def split_S2_images(s2_paths):
    '''
        It splits a list of Sentinel-2 path and returs two new lists: rgb images and cloud masks.
        The path must contains 'RGB' or 'CM'.
    '''

    cloud_mask = []
    rgb = []

    for p in s2_paths:
        if "RGB" in p:
            rgb.append(p)
        elif "CM" in p:
            cloud_mask.append(p)

    return rgb, cloud_mask

def reject_outliers(s1):
    p = np.percentile(s1, 99)
    s1 = np.clip(s1, 0.0, p)

    return s1

def get_s1_image(image_path, normalization):
    '''
        It returns an grayscale image (VV Sentinel-1) using a path.
        Image path must be the full path of an grayscale (VV) image (tif format)
    '''

    with rasterio.open(image_path) as src:
        vv = src.read()
        vv = np.moveaxis(vv,0,-1)

    #vv = 10 * np.log10(vv)
    vv = np.clip(vv, -25, 10)
    #vv = reject_outliers(vv)
    #Normalization in [-1, 1]
    vv = (vv - vv.min())/(vv.max() - vv.min())
    vv = np.clip(vv, 0.0, 1.0)
    
    return vv

def date_sort(e):
    '''
        This function is used to sort the paths by the months order.
    '''
    s = 0
    if 'Jan' in e:
        s = 0
    elif 'Feb' in e:
        s = 1
    elif 'Mar' in e:
        s = 2
    elif 'Apr' in e:
        s = 3
    elif 'May' in e:
        s = 4
    elif 'Jun' in e:
        s = 5
    elif 'Jul' in e:
        s = 6
    elif 'Aug' in e:
        s = 7
    elif 'Sep' in e:
        s = 8
    elif 'Oct' in e:
        s = 9
    elif 'Nov' in e:
        s = 10
    elif 'Dec' in e:
        s = 11

    return s

def patch_sort(e):

    '''
        This function is used to sort the paths by the patch order.
    '''

    s = 0
    if 'patch_0.tif' in e:
        s = 0
    elif 'patch_1.tif' in e:
        s = 1
    elif 'patch_2.tif' in e:
        s = 2
    elif 'patch_3.tif' in e:
        s = 3
    elif 'patch_4.tif' in e:
        s = 4
    elif 'patch_5.tif' in e:
        s = 5
    elif 'patch_6.tif' in e:
        s = 6
    elif 'patch_7.tif' in e:
        s = 7
    elif 'patch_8.tif' in e:
        s = 8
    elif 'patch_9.tif' in e:
        s = 9
    elif 'patch_10.tif' in e:
        s = 10
    elif 'patch_11.tif' in e:
        s = 11
    elif 'patch_12.tif' in e:
        s = 12
    elif 'patch_13.tif' in e:
        s = 13
    elif 'patch_14.tif' in e:
        s = 14
    elif 'patch_15.tif' in e:
        s = 15

    return s

def get_time_series(images):
    images.sort(key=patch_sort)
    series = []
    counter = 0
    for i in range(int(len(images)/4)):

        se = []

        for j in range(4):
            se.append(images[counter])
            se.sort(key=date_sort)
            counter = counter + 1

        for j in range(4):
            series.append(se[j])

    return series

def image_generatorLSTM(s2_paths, batch_size = 16, normalization='minmax', augment = True):
    batch_s2_input  = np.zeros((batch_size,3,256,256, 3))
    batch_output  = np.zeros((batch_size,256,256, 3))

    #print('Data Augmentation: {}'.format(augment))
    if augment:
        aug = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True)

    while True:
      for i in range(0, int(batch_size)):
        batch_index = randint(0, high=int(len(s2_paths)/4))*4

        if augment:
            transform = aug.get_random_transform(img_shape = (256,256,3))

        for j in range(4):
            s2 = get_s2_image(s2_paths[batch_index+j], normalization)

            if augment:
              s2 = aug.apply_transform(s2, transform)
          
            if j == 3:
                batch_output[i,:s2.shape[0],:s2.shape[1],...] = s2
            else:
                batch_s2_input[i,j,:s2.shape[0],:s2.shape[1],...] = s2

      yield batch_s2_input, batch_output


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray[:,:,np.newaxis]

def image_generatorCycleGAN(s2_paths, s1_paths, batch_size = 16, normalization='minmax', augment = True):

    batch_s2  = np.ones((batch_size, 256, 256, 3))
    batch_s1  = np.ones((batch_size, 256, 256, 3))

    #print('Data Augmentation: {}'.format(augment))
    if augment:
        aug = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True)

    while True:
        for i in range(0, int(batch_size)):

            batch_index = randint(0, high=int(len(s2_paths)/4))*4
            if augment:
                transform = aug.get_random_transform(img_shape = (256,256,1))

            s2 = get_s2_image(s2_paths[batch_index+3], normalization)
            #s1 = rgb2gray(s2)
            s1 = get_s1_image(s1_paths[batch_index+2], normalization)

            if augment:
                s2 = aug.apply_transform(s2, transform)
                s1 = aug.apply_transform(s1, transform)

           # s2 = tf.cast(s2, dtype=tf.float32)
           # s1 = tf.cast(s1, dtype=tf.float32)
            s2 = (2*s2) - 1.0
            s1 = (2*s1) - 1.0

            batch_s2[i,:s2.shape[0],:s2.shape[1],...] = s2
            batch_s1[i,:s1.shape[0],:s1.shape[1],0] = s1[...,0]
            batch_s1[i,:s1.shape[0],:s1.shape[1],1] = s1[...,0]
            batch_s1[i,:s1.shape[0],:s1.shape[1],2] = s1[...,0]
       

        yield batch_s2, batch_s1

def image_generatorHEAD(series, lstm, gan, batch_size = 16, normalization='minmax', augment = True):
    s2_paths = series[0]
    s1_paths = series[1]
    batch_input  = np.zeros((batch_size,256,256, 6))
    batch_output  = np.zeros((batch_size,256,256, 3))

    while True:
        for i in range(0, int(batch_size)):
            batch_index = randint(0, high=int(len(s2_paths)/4))*4

            batch_s2  = np.zeros((1, 3,256,256,3))
            batch_s1  = np.zeros((1, 256,256,3))

            for j in range(4):
                s2 = get_s2_image(s2_paths[batch_index+j], normalization)
                
                if j==3:
                    batch_output[i,:s2.shape[0],:s2.shape[1],...] = s2
                    s1 = get_s1_image(s1_paths[batch_index+j], normalization)
                    s1 = (2*s1) - 1.0
                    batch_s1[0, :s1.shape[0], :s1.shape[1], 0] = s1[...,0]
                    batch_s1[0, :s1.shape[0], :s1.shape[1], 1] = s1[...,0]
                    batch_s1[0, :s1.shape[0], :s1.shape[1], 1] = s1[...,0]
                else:
                    batch_s2[0, j, :s2.shape[0], :s2.shape[1],...] = s2

            lstm_o = lstm.predict(batch_s2)
            gan_o  = gan.predict(batch_s1)

            batch_input[i,:,:,:3] = lstm_o[0,-1,...,]
            batch_input[i,:,:,3:] = gan_o[0,...,]

        yield batch_input, batch_output
