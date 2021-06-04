from FBCSP import FBCSP
from EEGModels import EEGNet
from functions_dataset import *
from functions_network import *

from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf

if __name__ == "__main__":

    # TODO: nell'ablation considerare il caso di feature extraction
    # TODO: sistemare FBCSP e capire come funziona
    # TODO: implementare permutation per XAI
    # TODO: implementare saliency map

    data_dir = 'dataset/EEG'
    n_segments = 4

    fs = 250
    subjects = range(1, 10, 1)
    labels_name = {769: 'left', 770: 'right', 1: 'left', 2: 'right'}
    n_features = 4
    dataset, classes, dataset_features, classes_features = None, None, None, None

    for subject in subjects:

        if dataset is None:

            dataset, classes = load_dataset(data_dir, subjects[0])

            trials_dict = create_dict(dataset, classes, labels_name)
            FBCSP_f = FBCSP(trials_dict, fs, n_w=2, n_features=n_features, print_var=True)
            dataset_features, classes_features = FBCSP_f.createDataMatrix()

        else:

            d, c = load_dataset(data_dir, subject)
            dataset = np.concatenate((dataset, np.array(d)), axis=0)  # complete dataset
            classes = np.concatenate((classes, np.array(c)), axis=0)  # Labels {1, 2}

            trials_dict = create_dict(d, c, labels_name)
            FBCSP_f = FBCSP(trials_dict, fs, n_w=2, n_features=n_features, print_var=True)
            features, feat_label = FBCSP_f.createDataMatrix()
            # problema di diverse dimensioni date dalle dimensioni: non avviene considerando 1023
            dataset_features = np.concatenate((dataset_features, np.array(features)), axis=0)
            classes_features = np.concatenate((classes_features, np.array(feat_label)), axis=0)

    labels = []
    labels_features = []
    for i in range(len(classes)):
        labels.append([1, 0] if classes[i] == 1 else [0, 1])
    for i in range(len(classes_features)):
        labels_features.append([1, 0] if classes_features[i] == 1 else [0, 1])

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

    # USE OF EEGNET WITHOUT FEATURES

    model = training_EEGNet(train_dataset, train_labels, val_dataset, val_labels, batch_size, num_epochs,
                            'EEGNet_signal')
    # model = tf.keras.models.load_model('models/EEGNet_signal.h5')

    results = model.evaluate(test_dataset, test_labels, verbose=0)
    print("\nTest loss, Test accuracy: ", results)

    ablation(test_dataset, test_labels, model, results[1], n_segments=n_segments)

    # USE OF EEGNET WITH FFT

    train_fft = extract_FFT(train_dataset, n_segments)
    val_fft = extract_FFT(val_dataset, n_segments)
    test_fft = extract_FFT(test_dataset, n_segments)

    model = training_EEGNet(train_fft, train_labels, val_fft, val_labels, batch_size, num_epochs, 'EEGNet_fft')
    # model = tf.keras.models.load_model('models/EEGNet_fft.h5')

    results = model.evaluate(test_fft, test_labels, verbose=0)
    print("\nTest loss, Test accuracy: ", results)

    ablation(test_fft, test_labels, model, results[1])

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
    #
    # labels_features = np.array(labels_features)
    # print(dataset_features.shape)
    # print(labels_features.shape)
    #
    # train_data, val_data, train_labels, val_labels = train_test_split(dataset_features, labels_features,
    #                                                                   train_size=0.7, random_state=0)
    # val_data, test_data, val_labels, test_labels = train_test_split(val_data, val_labels, test_size=0.3,
    #                                                                 random_state=0)
    #
    # input_shape = (1, train_data[0].shape[0])
    #
    # model = EEGNet(nb_classes=2, Chans=1, Samples=input_shape[1])
    # model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    # history = model.fit(x=train_data, y=train_labels, batch_size=batch_size, epochs=num_epochs,
    #                     validation_data=(val_data, val_labels))
    # plot_model_training(history, 'model_FBCSP')
    # model.save('models/{}.h5'.format('model_FBCSP'))
    #
    # results = model.evaluate(test_data, test_labels, verbose=0)
    # print("\nTest loss, Test accuracy: ", results)
