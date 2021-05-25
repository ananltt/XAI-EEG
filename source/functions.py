from math import sqrt
import tensorflow as tf
from numpy.ma import count
from scipy.io import loadmat
import os
import numpy as np
from sklearn.preprocessing import scale
from reference import *


def DNN(shape):
    X_input = tf.keras.Input(shape)

    X = tf.keras.layers.Flatten()(X_input)
    X = tf.keras.layers.Dense(2, activation='softmax')(X)

    model_keras = tf.keras.Model(inputs=X_input, outputs=X)

    return model_keras

def CNN(input_shape):

    X_input = tf.keras.Input(input_shape)

    X = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same', name='conv0')(X_input)
    # X = tf.keras.layers.BatchNormalization(axis=-1, name='bn0')(X)
    X = tf.keras.layers.Activation('linear')(X)
    X = tf.keras.layers.Dropout(rate=0.9)(X)

    X = tf.keras.layers.MaxPool2D(pool_size=4)(X)

    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dense(2, activation='softmax', name='fc')(X)
    model = tf.keras.Model(inputs=X_input, outputs=X)

    return model


def load_dataset(ref, data_dir):

    dataset = []
    labels = []

    for row in range(len(ref)):

        participant_id = int(ref.iloc[int(row)]['subject'])
        trial = int(ref.iloc[int(row)]['trial'])
        left_label = int(ref.iloc[int(row)]['left_label'])
        right_label = int(ref.iloc[int(row)]['right_label'])

        if isinstance(data_dir, bytes):
            data_dir = data_dir.decode()
        if isinstance(participant_id, bytes):
            participant_id = participant_id.decode()

        file_name = str('S' + str(participant_id) + '_' + str(trial))
        file_dat = str(data_dir + '/' + str(file_name) + '.mat')
        data = loadmat(file_dat)['s']

        dataset.append(data)
        labels.append([left_label, right_label])

    return np.array(dataset), np.array(labels)
