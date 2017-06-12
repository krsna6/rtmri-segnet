#from keras import backend as K
from keras.layers.core import K
# setupo some things
K.set_image_dim_ordering('th')
TRAIN_MODEL = False
if not TRAIN_MODEL: K.set_learning_phase(0)
print('using ', K.image_dim_ordering())
import tensorflow as tf
import theano.tensor as T
SEED = 428
np.random.seed(SEED) # for reproducibility
from keras.datasets import mnist
from keras.layers.noise import GaussianNoise
import keras.models as models
from keras.layers import Layer, InputLayer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
#from keras.regularizers import ActivityRegularizer
#from keras.utils.visualize_util import plot
import os
import pylab as pl
import cv2
import numpy as np
import glob
from collections import Counter
import itertools
from functools import partial

# set class weights from some file - gt_path
n_classes = 14 #len(np.unique(gt_prior))
w = 84
h = 84
data_shape = 84*84

subj = 'yb1'

if TRAIN_MODEL:
    train_data = np.load('train_data_14labels_except_%s.npy' % (subj))
    train_label = np.load('train_label_14labels_except_%s.npy' % (subj))
    # doing only for the 12-nonzero classes
    train_data = train_data.reshape((len(train_data),1,w,h))
    if n_classes==12: train_label = train_label[:,:,1:]

test_data = np.load('test_data_14labels_yb1.npy')
test_label = np.load('test_label_14labels_yb1.npy')
test_data = test_data.reshape((len(test_data),1,w,h))
if n_classes==12: test_label = test_label[:,:,1:]

#- start weighted xent here
def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:,:, 0])
    y_pred_max = K.max(y_pred, axis=-1)
    y_pred_max = K.expand_dims(y_pred_max, axis=-1)
    y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
    for c_p, c_t in itertools.product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:,:, c_p] * y_true[:,:, c_t])
    #return K.mean( K.categorical_crossentropy(y_pred, y_true) * final_mask )
    return K.categorical_crossentropy(y_pred, y_true) * final_mask

# this is where I penalize heavily if one of my articulator-edges are classified as tissues
# first penalize only if edge is confused with air - tissue confusions leave the penalty to be uniform
# penalty parameter lambda_ = 0.5
lambda_ = 1.0
w_array = np.ones((14,14))
w_array[1:13, -1] = 1 + lambda_
ncce = partial(w_categorical_crossentropy, weights=w_array)
ncce.__name__ ='w_categorical_crossentropy'

#- end weighted xent here

class LRN2D(Layer):
    """
    This code is adapted from pylearn2.
    License at: https://github.com/lisa-lab/pylearn2/blob/master/LICENSE.txt
    """

    def __init__(self, alpha=1e-4, k=2, beta=0.75, n=5):
        if n % 2 == 0:
            raise NotImplementedError("LRN2D only works with odd n. n provided: " + str(n))
        super(LRN2D, self).__init__()
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n

    def get_output(self, train):
        X = self.get_input(train)
        b, ch, r, c = X.shape
        half_n = self.n // 2
        input_sqr = T.sqr(X)
        extra_channels = T.alloc(0., b, ch + 2*half_n, r, c)
        input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :], input_sqr)
        scale = self.k
        for i in range(self.n):
            scale += self.alpha * input_sqr[:, i:i+ch, :, :]
        scale = scale ** self.beta
        return X / scale

    def get_config(self):
        return {"name": self.__class__.__name__,
                "alpha": self.alpha,
                "k": self.k,
                "beta": self.beta,
                "n": self.n}


def create_encoding_layers(input_shape):
    kernel = 3
    filter_size = 16
    pad = 1
    pool_size = 3
    stride = (2,2)
    return [
        # ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(filter_size, kernel, kernel, border_mode='same', input_shape=input_shape),
        # BatchNormalization(),
        LRN2D(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size), strides=stride),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(32, kernel, kernel, border_mode='same'),
        # BatchNormalization(),
        LRN2D(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size), strides=stride),

        # ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(64, kernel, kernel, border_mode='same'),
        # BatchNormalization(),
        LRN2D(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size), strides=stride),

        # ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(128, kernel, kernel, border_mode='same'),
        # BatchNormalization(),
        LRN2D(),
        Activation('relu'),
        #MaxPooling2D(pool_size=(pool_size, pool_size)),
    ]

def create_decoding_layers():
    kernel = 3
    filter_size = 16
    pad = 1
    pool_size = 2
    return[
        #UpSampling2D(size=(pool_size,pool_size)),
        # ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(128, kernel, kernel, border_mode='same'),
        # BatchNormalization(),
        LRN2D(),

        UpSampling2D(size=(pool_size,pool_size)),
        # ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(64, kernel, kernel, border_mode='same'),
        # BatchNormalization(),
        LRN2D(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(32, kernel, kernel, border_mode='same'),
        # BatchNormalization(),
        LRN2D(),

        UpSampling2D(size=(pool_size,pool_size)),
        # ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(filter_size, kernel, kernel, border_mode='same'),
        # BatchNormalization(),
        LRN2D(),
    ]

'''Define these'''
nb_epoch = 20
batch_size = 250
model_name = 'b%s_ep%s_GT14_LOSOyb1_shuffled_custom_loss_LRN.hdf5' % (batch_size, nb_epoch)


if not TRAIN_MODEL:
    with tf.device('/cpu:0'):
        autoencoder = models.Sequential()
        # Add a noise layer to get a denoising autoencoder. This helps avoid overfitting
        #autoencoder.add(Dense(w,input_shape=(1,w,h)))
        #autoencoder.add(GaussianNoise(0.2))
        autoencoder.encoding_layers = create_encoding_layers( input_shape = (1,w,h) )
        autoencoder.decoding_layers = create_decoding_layers()
        for l in autoencoder.encoding_layers:
            autoencoder.add(l)
        for l in autoencoder.decoding_layers:
            autoencoder.add(l)


        autoencoder.add(Convolution2D(n_classes, 1, 1, border_mode='valid',))
        autoencoder.add(Reshape((n_classes,data_shape)))
        autoencoder.add(Permute((2, 1)))
        autoencoder.add(Activation('softmax'))
        autoencoder.compile(loss=ncce, optimizer='adadelta')

        autoencoder.load_weights('model_weight_'+model_name)

else:
    autoencoder = models.Sequential()
    # Add a noise layer to get a denoising autoencoder. This helps avoid overfitting
    #autoencoder.add(GaussianNoise(0.2))
    autoencoder.encoding_layers = create_encoding_layers( input_shape = (1,w,h) )
    autoencoder.decoding_layers = create_decoding_layers()
    for l in autoencoder.encoding_layers:
        autoencoder.add(l)
    for l in autoencoder.decoding_layers:
        autoencoder.add(l)

    autoencoder.add(Convolution2D(n_classes, 1, 1, border_mode='valid', ))
    autoencoder.add(Reshape((n_classes, data_shape)))
    autoencoder.add(Permute((2, 1)))
    autoencoder.add(Activation('softmax'))
    autoencoder.compile(loss=ncce, optimizer='adadelta')

    history = autoencoder.fit(train_data, train_label, batch_size=batch_size, nb_epoch=nb_epoch,
                                      verbose=1, shuffle=True, validation_data=(test_data,test_label))
                                      #validation_split=0.2)  # , class_weight=class_weights)

    autoencoder.save('model_'+model_name)
    autoencoder.save_weights('model_weight_'+model_name)
