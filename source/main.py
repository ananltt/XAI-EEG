import sys
from functions_dataset import *
from functions_network import *
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import os

if __name__ == "__main__":

    sys.stdout = open("../output/output.txt", "w")  # TO WRITE ALL OUTPUT IN A FILE

    data_dir = '../dataset/EEG'

    n_segments = 8              # number of segments considered in the signal
    n_features = 396            # number of features for FBCSP
    fs = 250                    # sampling frequency

    subjects = range(1, 10, 1)  # dataset composition
    dataset, labels = None, None

    for subject in subjects:

        if dataset is None:
            dataset, labels = load_dataset(data_dir, subject, consider_artefacts=False)

        else:
            d, l = load_dataset(data_dir, subject, consider_artefacts=False)
            dataset = np.concatenate((dataset, np.array(d)), axis=0)  # complete dataset
            labels = np.concatenate((labels, np.array(l)), axis=0)

    # Common hyperparameters for the training

    batch_size = 32
    num_epochs = 50

    labels = np.array(labels)
    train_dataset, val_dataset, train_labels, val_labels = train_test_split(dataset, labels, train_size=0.7,
                                                                            random_state=1)
    val_dataset, test_dataset, val_labels, test_labels = train_test_split(val_dataset, val_labels, train_size=0.7,
                                                                          random_state=1)

    wavelet_variation(train_dataset[0][0])

    # USE OF EEGNET WITHOUT FEATURES

    print("\nSIGNAL DATASET:\n")

    if not os.path.exists('../models/EEGNet_signal.h5'):
        model = training_EEGNet(train_dataset, train_labels, val_dataset, val_labels, batch_size, num_epochs,
                                '../models/EEGNet_signal')
    else:
        model = tf.keras.models.load_model('../models/EEGNet_signal.h5')

    results = model.evaluate(test_dataset, test_labels, verbose=0)
    print("\nTest loss, Test accuracy: ", results)

    ablation(test_dataset, test_labels, model, n_segments=n_segments)
    ablation_label_depending(test_dataset, test_labels, model, n_segments=n_segments)

    permutation(test_dataset, test_labels, model, n_segments=n_segments)

    # USE OF EEGNET WITH WAVELET

    print("\nWAVELET DATASET:\n")

    train_wt = extract_wt(train_dataset)
    val_wt = extract_wt(val_dataset)
    test_wt = extract_wt(test_dataset)

    if not os.path.exists('../models/EEGNet_wt.h5'):
        model = training_EEGNet(train_wt, train_labels, val_wt, val_labels, batch_size, num_epochs, '../models/EEGNet_wt')
    else:
        model = tf.keras.models.load_model('../models/EEGNet_wt.h5')

    results = model.evaluate(test_wt, test_labels, verbose=0)
    print("\nTest loss, Test accuracy: ", results)

    ablation(test_dataset, test_labels, model, extract_wt, n_segments)
    ablation_label_depending(test_dataset, test_labels, model, extract_wt, n_segments)

    permutation(test_dataset, test_labels, model, extract_wt, n_segments)

    # USE OF EEGNET WITH FBCSP

    print("\nFBCSP DATASET:\n")

    train_fbcsp = extractFBCSP(train_dataset, train_labels, n_features)
    val_fbcsp = extractFBCSP(val_dataset, val_labels, n_features)
    test_fbcsp = extractFBCSP(test_dataset, test_labels, n_features)

    if not os.path.exists('../models/EEGNet_FBCSP.h5'):
        model = training_EEGNet(train_fbcsp, train_labels, val_fbcsp, val_labels, batch_size, num_epochs,
                                '../models/EEGNet_FBCSP')
    else:
        model = tf.keras.models.load_model('../models/EEGNet_FBCSP.h5')

    results = model.evaluate(test_fbcsp, test_labels, verbose=0)
    print("\nTest loss, Test accuracy: ", results)

    # ablation(test_dataset, test_labels, model, extractFBCSP, n_segments, n_features=n_features)
    # ablation_label_depending(test_dataset, test_labels, model, extractFBCSP, n_segments, n_features=n_features)
    # permutation(test_dataset, test_labels, model, extractFBCSP, n_segments, n_features=n_features)

    sys.stdout.close()
