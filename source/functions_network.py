import copy
import numpy as np
from EEGModels import EEGNet
from matplotlib import pyplot as plt
from functions_dataset import extract_indexes_segments


def training_EEGNet(train_data, train_labels, val_data, val_labels, batch_size, num_epochs, model_path):
    """
    Function for the training of an EEGNet. In general, have been used categorical_crossentropy as loss function, adam
    optimizer and accuracy as metric. The trained model is then saved and loss and accuracy are plotted.

    :param train_data: training dataset (n.trials x n.channels x n.samples)
    :param train_labels: training labels ([0, 1] for right hand, [1, 0] for left hand: n.trials x 2)
    :param val_data: validation dataset
    :param val_labels: validation labels
    :param batch_size: batch size for training
    :param num_epochs: number of epochs for training
    :param model_path: path and model name
    :return: trained model
    """

    input_shape = (train_data[0].shape[0], train_data[0].shape[1])

    model = EEGNet(nb_classes=2, Chans=input_shape[0], Samples=input_shape[1])
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    history = model.fit(x=train_data, y=train_labels, batch_size=batch_size, epochs=num_epochs,
                        validation_data=(val_data, val_labels), verbose=2)

    plot_model_training(history, model_path)
    model.save('{}.h5'.format(model_path))

    return model


def plot_model_training(history, model_name):
    """
    Plot of loss and accuracy in training and validation tests during the training of a model

    :param history: history of the model training
    :param model_name: name of the model (will be name of the saved image)
    """

    # accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('{}_accuracy.png'.format(model_name))
    plt.close()

    # loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('{}_loss.png'.format(model_name))
    plt.close()


def ablation(dataset, labels, model, function_features=None, n_segments=4, n_channels=22, n_features=396):
    """
    Function to perform different types of ablation according to the XAI definition

    :param dataset: dataset on which evaluate the ablation (if a feature dataset must be evaluated, this dataset should
    be without feature extraction)
    :param labels: labels corresponding to the dataset
    :param model: model on which evaluate the ablation
    :param function_features: function for the feature extraction of the dataset
    :param n_segments: number of segments to be evaluated with XAI
    :param n_channels: number of channels to be evaluated with XAI
    :param n_features: number of features to be extracted from FBCSP
    """

    accuracies = ablation_zero_segments(dataset, labels, model, function_features, n_segments, n_features)
    print("\nAblation with zeros in the segments: \n", accuracies)

    accuracies = ablation_linear_segments(dataset, labels, model, function_features, n_segments, n_features)
    print("\nAblation with linearity in the segments: \n", accuracies)

    accuracies = ablation_zero_channels(dataset, labels, model, function_features, n_channels, n_features)
    print("\nAblation with zeros in the channels: \n", accuracies)


def ablation_label_depending(dataset, labels, model, function_features=None, n_segments=4, n_channels=22,
                             n_features=396):
    """
    Function to perform ablation separately for each label present in the dataset

    :param dataset: dataset on which evaluate the ablation (if a feature dataset must be evaluated, this dataset should
    be without feature extraction)
    :param dataset: dataset on which evaluate the ablation (if a feature dataset must be evaluated, this dataset should
    be without feature extraction)
    :param labels: labels corresponding to the dataset
    :param model: model on which evaluate the ablation
    :param function_features: function for the feature extraction of the dataset
    :param n_segments: number of segments to be evaluated with XAI
    :param n_channels: number of channels to be evaluated with XAI
    :param n_features: number of features to be extracted with FBCSP
    """

    # Extract the unique labels

    classes, indexes = np.unique(labels, return_inverse=True, axis=0)

    for j, c in enumerate(classes):

        print("\nConsidering labels {}".format(c))

        # Build the dataset corresponding to each label, in case applying the feature extractor algorithm

        data = np.array([dataset[i] for i in range(len(indexes)) if indexes[i] == j])
        lab = np.repeat([c], data.shape[0], axis=0)

        if function_features is not None:
            if function_features.__name__ == "extractFBCSP":
                x = function_features(data, lab, n_features)
            else:
                x = function_features(data)
        else:
            x = data

        # Evaluate the model with the built dataset

        results = model.evaluate(x, lab, verbose=0)
        print("\nTest loss, Test accuracy: ", results)

        # Perform ablation with the built dataset
        ablation(data, lab, model, function_features, n_segments, n_channels, n_features)


def ablation_zero_segments(dataset, labels, model, function_features=None, n_segments=4, n_features=396):
    """
    Function to perform ablation setting at zero the segment of the signal under investigation

    :param dataset: dataset on which evaluate the ablation (if a feature dataset must be evaluated, this dataset should
    be without feature extraction)
    :param labels: labels corresponding to the dataset
    :param model: model on which evaluate the ablation
    :param function_features: function for the feature extraction of the dataset
    :param n_segments: number of segments to be evaluated with XAI
    :param n_features: number of features to be extracted with FBCSP
    :return: each value of the array represents the accuracy obtained without the corresponding segment
    """

    # Extract the indices of each segment

    indexes = extract_indexes_segments(dataset.shape[2], n_segments)
    accuracies = np.empty(n_segments)

    for k in range(n_segments):

        data = copy.deepcopy(dataset)
        start, end = indexes[k]

        # Set at zero the segment under investigation and re-build the dataset

        data[:, :, start:end] = np.zeros((data.shape[0], data.shape[1], end - start))

        if function_features is not None:
            if function_features.__name__ == "extractFBCSP":
                x = function_features(dataset, labels, n_features)
            else:
                x = function_features(data)
        else:
            x = data

        # Evaluate difference of accuracy

        results = model.evaluate(x, labels, verbose=0)
        accuracies[k] = results[1]

    return accuracies


def ablation_linear_segments(dataset, labels, model, function_features=None, n_segments=4, n_features=396):
    """
    Function to perform ablation setting the segment under investigation with a linear function

    :param dataset: dataset on which evaluate the ablation (if a feature dataset must be evaluated, this dataset should
    be without feature extraction)
    :param labels: labels corresponding to the dataset
    :param model: model on which evaluate the ablation
    :param function_features: function for the feature extraction of the dataset
    :param n_segments: number of segments to be evaluated with XAI
    :param n_features: number of features to be extracted with FBCSP
    :return: each value of the array represents the accuracy obtained without the corresponding segment
    """

    # Extract the indices of the segments

    indexes = extract_indexes_segments(dataset.shape[2], n_segments)
    accuracies = np.empty(n_segments)

    for k in range(n_segments):

        data = copy.deepcopy(dataset)
        start, end = indexes[k]

        # For each channel inside each trial, substitute the segment under investigation with a linear function (between
        # the extremities) and rebuild the dataset

        for j in range(data.shape[0]):
            for i in range(data.shape[1]):
                prev = data[j, i, start]
                cons = data[j, i, end - 1]
                lin_len = end - start

                lin = np.linspace(prev, cons, num=lin_len)
                data[j, i, start:end] = lin

        if function_features is not None:
            if function_features.__name__ == "extractFBCSP":
                x = function_features(dataset, labels, n_features)
            else:
                x = function_features(data)
        else:
            x = data

        # Evaluate the difference of accuracies

        results = model.evaluate(x, labels, verbose=0)
        accuracies[k] = results[1]

    return accuracies


def ablation_zero_channels(dataset, labels, model, function_features=None, n_channels=22, n_features=396):
    """
    Function to perform ablation setting the signal from the channel under investigation at zero

    :param dataset: dataset on which evaluate the ablation (if a feature dataset must be evaluated, this dataset should
    be without feature extraction)
    :param labels: labels corresponding to the dataset
    :param model: model on which evaluate the ablation
    :param function_features: function for the feature extraction of the dataset
    :param n_channels: number of channels to be evaluated with XAI
    :param n_features: number of features to be extracted with FBCSP
    :return: each value of the array represents the accuracy obtained without the corresponding segment
    """

    accuracies = np.empty(n_channels)

    for k in range(n_channels):

        data = copy.deepcopy(dataset)

        # Set the channel values of zero and rebuild the dataset

        data[:, k, :] = np.zeros((data.shape[0], data.shape[2]))

        if function_features is not None:
            if function_features.__name__ == "extractFBCSP":
                x = function_features(dataset, labels, n_features)
            else:
                x = function_features(data)
        else:
            x = data

        # Evaluate the difference of accuracies

        results = model.evaluate(x, labels, verbose=0)
        accuracies[k] = results[1]

    return accuracies


def permutation(dataset, labels, model, function_features=None, n_segments=4, n_channels=22,
                n_features=396):
    """
    Function to perform different types of permutation according to the XAI definition

    :param dataset: dataset on which evaluate the ablation (if a feature dataset must be evaluated, this dataset should
    be without feature extraction)
    :param labels: labels corresponding to the dataset
    :param model: model on which evaluate the ablation
    :param function_features: function for the feature extraction of the dataset
    :param n_segments: number of segments to be evaluated with XAI
    :param n_channels: number of channels to be evaluated with XAI
    :param n_features: number of features to be extracted with FBCSP
    """

    accuracies = permutation_segments(dataset, labels, model, function_features, n_segments, n_features)
    print("\nPermutation in the segments: \n", accuracies)

    accuracies = ablation_zero_channels(dataset, labels, model, function_features, n_channels, n_features)
    print("\nPermutation in the channels: \n", accuracies)


def permutation_segments(dataset, labels, model, function_features=None, n_segments=4, n_features=396):
    """
    Function to perform permutation substituting a segment of a channel of a trial with the same segment of the same
    channel of another trial

    :param dataset: dataset on which evaluate the ablation (if a feature dataset must be evaluated, this dataset should
    be without feature extraction)
    :param labels: labels corresponding to the dataset
    :param model: model on which evaluate the ablation
    :param function_features: function for the feature extraction of the dataset
    :param n_segments: number of segments to be evaluated with XAI
    :param n_features: number of features to be extracted with FBCSP
    :return: each value of the array represents the accuracy obtained without the corresponding segment
    """

    # Extract the indexes of the segments

    indexes = extract_indexes_segments(dataset.shape[2], n_segments)
    accuracies = np.empty(n_segments)

    for k in range(n_segments):

        data = copy.deepcopy(dataset)
        start, end = indexes[k]

        # Extract the random new trial, substitute the segment and rebuild the dataset

        list_trials = range(data.shape[0])

        for i in range(data.shape[0]):
            # list_channels = range(data.shape[1])
            for j in range(data.shape[1]):
                actual_trials = [t for t in list_trials if t != i]
                p = np.random.choice(actual_trials)
                # actual_channels = [c for c in list_channels if c != j]
                # q = np.random.choice(actual_channels)
                data[i, j, start:end] = data[p, j, start:end]

        if function_features is not None:
            if function_features.__name__ == "extractFBCSP":
                x = function_features(dataset, labels, n_features)
            else:
                x = function_features(data)
        else:
            x = data

        # Evaluate the difference of accuracy

        results = model.evaluate(x, labels, verbose=0)
        accuracies[k] = results[1]

    return accuracies


def permutation_channels(dataset, labels, model, function_features=None, n_channels=22, n_features=396):
    """
    Function to perform permutation substituting a channel with the same channel of another trial

    :param dataset: dataset on which evaluate the ablation (if a feature dataset must be evaluated, this dataset should
    be without feature extraction)
    :param labels: labels corresponding to the dataset
    :param model: model on which evaluate the ablation
    :param function_features: function for the feature extraction of the dataset
    :param n_channels: number of channels to be evaluated with XAI
    :param n_features: number of features to be extracted with FBCSP
    :return: each value of the array represents the accuracy obtained without the corresponding segment
    """

    accuracies = np.empty(n_channels)

    for k in range(n_channels):

        data = copy.deepcopy(dataset)
        list_trials = range(data.shape[0])

        # Select the random new trial, substitute the channel and rebuild the dataset

        for i in range(data.shape[0]):
            actual_trials = [t for t in list_trials if t != i]
            p = np.random.choice(actual_trials)
            data[i, k, :] = data[p, k, :]

        if function_features is not None:
            if function_features.__name__ == "extractFBCSP":
                x = function_features(dataset, labels, n_features)
            else:
                x = function_features(data)
        else:
            x = data

        # Evaluate the difference of accuracies

        results = model.evaluate(x, labels, verbose=0)
        accuracies[k] = results[1]

    return accuracies
