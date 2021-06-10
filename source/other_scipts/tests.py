import sys
from ..functions_network import *
from ..functions_dataset import *
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import os

if __name__ == "__main__":

    sys.stdout = open("../output/output_test.txt", "w")  # TO WRITE ALL OUTPUT IN A FILE

    data_dir = '../dataset/EEG'
    n_segments = 8              # number of segments considered in the signal
    n_features = 396            # number of features for FBCSP
    fs = 250                    # sampling frequency

    subjects = range(1, 10, 1)  # dataset composition
    dataset, labels = None, None

    for subject in subjects:

        if dataset is None:
            dataset, labels = load_dataset(data_dir, subject)

        else:
            d, l = load_dataset(data_dir, subject)
            dataset = np.concatenate((dataset, np.array(d)), axis=0)  # complete dataset
            labels = np.concatenate((labels, np.array(l)), axis=0)  # Labels {1, 2}

    # Common hyperparameters for the training

    batch_size = 16
    num_epochs = 50

    labels = np.array(labels)
    train_dataset, val_dataset, train_labels, val_labels = train_test_split(dataset, labels, train_size=0.7,
                                                                            random_state=0)
    val_dataset, test_dataset, val_labels, test_labels = train_test_split(val_dataset, val_labels, train_size=0.7,
                                                                          random_state=0)

    train_steps = int(np.ceil(train_dataset.shape[0] / batch_size))
    val_steps = int(np.ceil(val_dataset.shape[0] / batch_size))
    test_steps = int(np.ceil(test_dataset.shape[0] / batch_size))

    # USE OF EEGNET WITH SC

    print("\nSTATISTICAL CHARACTERISTICS DATASET:\n")

    train_sc = extract_statistical_characteristics(train_dataset)
    val_sc = extract_statistical_characteristics(val_dataset)
    test_sc = extract_statistical_characteristics(test_dataset)

    if not os.path.exists('../models/EEGNet_sc.h5'):
        model = training_EEGNet(train_sc, train_labels, val_sc, val_labels, batch_size, num_epochs, 'EEGNet_sc')
    else:
        model = tf.keras.models.load_model('../models/EEGNet_sc.h5')

    results = model.evaluate(test_sc, test_labels, verbose=0)
    print("\nTest loss, Test accuracy: ", results)

    ablation(test_dataset, test_labels, model, extract_statistical_characteristics, n_segments)
    ablation_label_depending(test_dataset, test_labels, model, extract_statistical_characteristics, n_segments)

    permutation(test_dataset, test_labels, model, extract_statistical_characteristics, n_segments)

    # USE OF EEGNET WITH PSD

    print("\nPSD DATASET:\n")

    train_psd = extract_psd(train_dataset)
    val_psd = extract_psd(val_dataset)
    test_psd = extract_psd(test_dataset)

    if not os.path.exists('../models/EEGNet_psd.h5'):
        model = training_EEGNet(train_psd, train_labels, val_psd, val_labels, batch_size, num_epochs, 'EEGNet_psd')
    else:
        model = tf.keras.models.load_model('../models/EEGNet_psd.h5')

    results = model.evaluate(test_psd, test_labels, verbose=0)
    print("\nTest loss, Test accuracy: ", results)

    ablation(test_dataset, test_labels, model, extract_psd, n_segments)
    ablation_label_depending(test_dataset, test_labels, model, extract_psd, n_segments)

    permutation(test_dataset, test_labels, model, extract_psd, n_segments)

    sys.stdout.close()
