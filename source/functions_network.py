import copy
import numpy as np
from EEGModels import EEGNet
from matplotlib import pyplot as plt
from functions_dataset import extract_indexes_segments


def training_EEGNet(train_data, train_labels, val_data, val_labels, batch_size, num_epochs, model_name):
    """
    Function for the training of an EEGNet. In general, have been used categorical_crossentropy as loss function, adam
    optimizer and accuracy as metric. The trained model is then saved and loss and accuracy are plotted.

    :param train_data: training dataset (n.trials x n.channels x n.samples)
    :param train_labels: training labels ([0, 1] for right hand, [1, 0] for left hand: n.trials x 2)
    :param val_data: validation dataset
    :param val_labels: validation labels
    :param batch_size: batch size for training
    :param num_epochs: number of epochs for training
    :param model_name: model name
    :return: trained model
    """

    input_shape = (train_data[0].shape[0], train_data[0].shape[1])

    model = EEGNet(nb_classes=2, Chans=input_shape[0], Samples=input_shape[1])
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    history = model.fit(x=train_data, y=train_labels, batch_size=batch_size, epochs=num_epochs,
                        validation_data=(val_data, val_labels), verbose=2)

    plot_model_training(history, model_name)
    model.save('models/{}.h5'.format(model_name))

    return model


def plot_model_training(history, model_name):
    """
    Plot of loss and accuracy in training and validation tests during the training of a model

    :param history: history of the model training
    :param model_name: name of the model (will be name of the saved image)
    :return:
    """

    # accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('output/model_accuracy_{}.png'.format(model_name))
    plt.close()

    # loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('output/model_loss_{}.png'.format(model_name))
    plt.close()


def ablation(dataset, labels, model, total_accuracy, function_features=None, n_segments=4, n_channels=22):

    differences = ablation_zero_segments(dataset, labels, model, total_accuracy, function_features, n_segments)
    print("\nAblation with zeros in the segments: \n", differences)

    differences = ablation_linear_segments(dataset, labels, model, total_accuracy, function_features, n_segments)
    print("\nAblation with linearity in the segments: \n", differences)

    differences = ablation_zero_channels(dataset, labels, model, total_accuracy, function_features, n_channels)
    print("\nAblation with zeros in the channels: \n", differences)


def ablation_label_depending(dataset, labels, model, function_features=None, n_segments=4, n_channels=22, n_features=36):

    classes, indexes = np.unique(labels, return_inverse=True, axis=0)

    for j, c in enumerate(classes):

        print("\nConsidering labels {}".format(c))

        data = np.array([dataset[i] for i in range(len(indexes)) if indexes[i] == j])
        lab = np.repeat([c], data.shape[0], axis=0)

        if function_features is not None:
            if function_features.__name__ == "extractFBCSP":
                x = function_features(data, lab, n_features)
            else:
                x = function_features(data)
        else:
            x = data

        results = model.evaluate(x, lab, verbose=0)

        ablation(data, lab, model, results[1], function_features, n_segments, n_channels)


def ablation_zero_segments(dataset, labels, model, accuracy, function_features=None, n_segments=4, n_features=36):

    differences = np.empty(n_segments)
    indexes = extract_indexes_segments(dataset.shape[2], n_segments)

    for k in range(n_segments):

        data = copy.deepcopy(dataset)
        start, end = indexes[k]

        data[:, :, start:end] = np.zeros((data.shape[0], data.shape[1], end - start))

        if function_features is not None:
            if function_features.__name__ == "extractFBCSP":
                x = function_features(dataset, labels, n_features)
            else:
                x = function_features(data)
        else:
            x = data

        results = model.evaluate(x, labels, verbose=0)
        differences[k] = accuracy - results[1]

    return differences


def ablation_linear_segments(dataset, labels, model, accuracy, function_features=None, n_segments=4, n_features=36):
    differences = np.empty(n_segments)
    indexes = extract_indexes_segments(dataset.shape[2], n_segments)

    for k in range(n_segments):

        data = copy.deepcopy(dataset)
        start, end = indexes[k]

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

        results = model.evaluate(x, labels, verbose=0)
        differences[k] = accuracy - results[1]

    return differences


def ablation_zero_channels(dataset, labels, model, accuracy, function_features=None, n_channels=22, n_features=22):
    differences = np.empty(n_channels)

    for k in range(n_channels):

        data = copy.deepcopy(dataset)

        data[:, k, :] = np.zeros((data.shape[0], data.shape[2]))

        if function_features is not None:
            if function_features.__name__ == "extractFBCSP":
                x = function_features(dataset, labels, n_features)
            else:
                x = function_features(data)
        else:
            x = data

        results = model.evaluate(x, labels, verbose=0)
        differences[k] = accuracy - results[1]

    return differences


def permutation(dataset, labels, model, total_accuracy, function_features=None, n_segments=4, n_channels=22):

    differences = permutation_segments(dataset, labels, model, total_accuracy, function_features, n_segments)
    print("\nPermutation in the segments: \n", differences)

    differences = ablation_zero_channels(dataset, labels, model, total_accuracy, function_features, n_channels)
    print("\nPermutation in the channels: \n", differences)


def permutation_segments(dataset, labels, model, accuracy, function_features=None, n_segments=4, n_features=36):

    differences = np.empty(n_segments)
    indexes = extract_indexes_segments(dataset.shape[2], n_segments)

    for k in range(n_segments):

        data = copy.deepcopy(dataset)
        start, end = indexes[k]
        list_trials = range(data.shape[0])

        for i in range(data.shape[0]):
            actual_trials = [t for t in list_trials if t != i]
            p = np.random.choice(actual_trials)
            data[i, :, start:end] = data[p, :, start:end]

        if function_features is not None:
            if function_features.__name__ == "extractFBCSP":
                x = function_features(dataset, labels, n_features)
            else:
                x = function_features(data)
        else:
            x = data

        results = model.evaluate(x, labels, verbose=0)
        differences[k] = accuracy - results[1]

    return differences


def permutation_channels(dataset, labels, model, accuracy, function_features=None, n_channels=22, n_features=36):

    differences = np.empty(n_channels)

    for k in range(n_channels):

        data = copy.deepcopy(dataset)
        list_trials = range(data.shape[0])

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

        results = model.evaluate(x, labels, verbose=0)
        differences[k] = accuracy - results[1]

    return differences
