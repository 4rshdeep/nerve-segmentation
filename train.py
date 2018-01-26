from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K

from skimage.transform import resize
from skimage.io import imsave

import numpy as np
import os

from data import load_train_data, load_test_data

K.set_image_data_format('channels_last')  #tf dimension order

smooth = 1.
img_rows = 96
img_cols = 96

# Implementing Sorensons formula
# https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
def dice_coef(y_true, y_pred):
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersection = K.sum(y_true_flat * y_pred_flat)
    return (2. * intersection + smooth)/(K.sum(y_true_flat) + K.sum(y_pred_flat) + smooth)

# losses source
# https://github.com/keras-team/keras/blob/master/keras/losses.py
def dice_loss(y_true, y_pred):
    # higher dice -> more similar 
    return -dice_coef(y_true, y_pred)

def get_unet():
    """

    For convolution 
    O = (I - F + 2*P)*S + 1

    For Deconvolution
    O = S*(I-1) +F -2*P

    """
    inputs = Input((img_rows, img_cols, 1))
    #96, 96, 1
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    #96, 96, 32
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #48, 48, 32

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    #48, 48, 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #24, 24, 64

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    #24, 24, 128
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #12, 12, 128

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    #12, 12, 256
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    #6, 6, 256

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    #6, 6, 512

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    # stacking two 12, 12, 256 -> 12, 12, 512
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    # 12, 12, 256 
    
    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    # stacking two 24, 24, 128 -> 24, 24, 256
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    # 24, 24, 128
    
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    # stacking two 48, 48, 64 -> 48, 48, 128
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    # 48, 48, 64

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    # stacking two 96, 96, 32 -> 96, 96, 64
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
    # 96, 96, 32

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    # 96, 96, 1
    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_loss, metrics=[dice_coef])

    return model

def preprocess(imgs):
    imgs_processed = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    
    for i in range(imgs.shape[0]):
        imgs_processed[i] = resize(imgs[i], (img_rows, img_cols), preserve_range=True)
    
    imgs_p = imgs_processed[..., np.newaxis] # equivalent to imgs_p[:, :, :, np.newaxis]
    return imgs_p

def train_and_predict():
    print("-"*30)
    print("Loading and preprocessing train data")
    print("-"*30)

    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    imgs_mask_train = imgs_mask_train.astype('float32')

    mean = np.mean(imgs_train)
    std = np.std(imgs_train)

    # normalise
    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train /= 255.

    print("-"*30)
    print("Creating and compiling model...")
    print("-"*30)

    model = get_unet()

    checkpoint = ModelCheckpoint(filepath='weights.hdf5', monitor='val_loss', save_best_only=True)

    print("-"*30)
    print("Fitting Model")
    print("-"*30)

    # model.fit(imgs_train, imgs_mask_train, batch_size=32, epochs=20, verbose=1, shuffle=True,
    #           validation_split=0.2,
    #           callbacks=[checkpoint])

    print("-"*30)
    print("Loading and preprocessing test data ...")
    print("-"*30)
    
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print("-"*30)
    print("Loading saved weights ...")
    print("-"*30)
    model.load_weights('weights.hdf5')
    
    print("-"*30)
    print("Predicting on test data ...")
    print("-"*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)

    print("-"*30)
    print("Saving predicted masks to files ...")
    print("-"*30)

    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png') , image)

if __name__ == '__main__':
    train_and_predict()