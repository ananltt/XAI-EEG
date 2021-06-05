from FBCSP import FBCSP
from EEGModels import EEGNet
from functions_dataset import *
from functions_network import *
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

from source.FBCSP_V4 import FBCSP_V4

if __name__ == "__main__":

    # TODO: nell'ablation considerare il caso di feature extraction -> fatto ma valori sempre uguali
    # TODO: sistemare FBCSP e capire come funziona
    # TODO: implementare saliency map

    data_dir = 'dataset/EEG'
    n_segments = 4
    n_features = 22

    fs = 250
    subjects = range(1, 10, 1)

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
    num_epochs = 40

    labels = np.array(labels)
    train_dataset, val_dataset, train_labels, val_labels = train_test_split(dataset, labels, train_size=0.7,
                                                                            random_state=0)
    val_dataset, test_dataset, val_labels, test_labels = train_test_split(val_dataset, val_labels, train_size=0.7,
                                                                          random_state=0)

    train_steps = int(np.ceil(train_dataset.shape[0] / batch_size))
    val_steps = int(np.ceil(val_dataset.shape[0] / batch_size))
    test_steps = int(np.ceil(test_dataset.shape[0] / batch_size))

    # # USE OF EEGNET WITHOUT FEATURES
    #
    # # model = training_EEGNet(train_dataset, train_labels, val_dataset, val_labels, batch_size, num_epochs,
    # #                         'EEGNet_signal')
    # model = tf.keras.models.load_model('models/EEGNet_signal.h5')
    #
    # results = model.evaluate(test_dataset, test_labels, verbose=0)
    # print("\nTest loss, Test accuracy: ", results)
    #
    # ablation(test_dataset, test_labels, model, results[1], n_segments=n_segments)
    # ablation_label_depending(test_dataset, test_labels, model, n_segments=n_segments)
    #
    # permutation(test_dataset, test_labels, model, results[1], n_segments=n_segments)

    # # USE OF EEGNET WITH FFT
    #
    # train_fft = extract_FFT(train_dataset, n_segments)
    # val_fft = extract_FFT(val_dataset, n_segments)
    # test_fft = extract_FFT(test_dataset, n_segments)
    #
    # # model = training_EEGNet(train_fft, train_labels, val_fft, val_labels, batch_size, num_epochs, 'EEGNet_fft')
    # model = tf.keras.models.load_model('models/EEGNet_fft.h5')
    #
    # results = model.evaluate(test_fft, test_labels, verbose=0)
    # print("\nTest loss, Test accuracy: ", results)
    #
    # ablation(test_dataset, test_labels, model, results[1], extract_wt, n_segments)

    # # USE OF EEGNET WITH WAVELET
    #
    # train_wt = extract_wt(train_dataset)
    # val_wt = extract_wt(val_dataset)
    # test_wt = extract_wt(test_dataset)
    #
    # model = training_EEGNet(train_fft, train_labels, val_fft, val_labels, batch_size, num_epochs, 'EEGNet_wt')
    # # model = tf.keras.models.load_model('models/EEGNet_wt.h5')
    #
    # results = model.evaluate(test_wt, test_labels, verbose=0)
    # print("\nTest loss, Test accuracy: ", results)
    #
    # ablation(test_wt, test_labels, model, results[1])

    # USE OF EEGNET WITH FBCSP

    train_fbcsp = extractFBCSP(train_dataset, train_labels, n_features)
    val_fbcsp = extractFBCSP(val_dataset, val_labels, n_features)
    test_fbcsp = extractFBCSP(test_dataset, test_labels, n_features)

    model = training_EEGNet(train_fbcsp, train_labels, val_fbcsp, val_labels, batch_size, num_epochs,
                            'EEGNet_FBCSP')

    # results = model.evaluate(test_data, test_labels, verbose=0)
    # print("\nTest loss, Test accuracy: ", results)
