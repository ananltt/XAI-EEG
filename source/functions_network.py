import copy
import numpy as np
from EEGModels import EEGNet
from matplotlib import pyplot as plt


def training_EEGNet(train_data, train_labels, val_data, val_labels, batch_size, num_epochs, model_name):

    input_shape = (train_data[0].shape[0], train_data[0].shape[1], 1)

    model = EEGNet(nb_classes=2, Chans=input_shape[0], Samples=input_shape[1])
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    history = model.fit(x=train_data, y=train_labels, batch_size=batch_size, epochs=num_epochs,
                        validation_data=(val_data, val_labels))
    plot_model_training(history, model_name)
    model.save('models/{}.h5'.format(model_name))

    return model


def plot_model_training(history, model_name):

    # accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('images/model_accuracy_{}.png'.format(model_name))
    plt.close()

    # loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('images/model_loss_{}.png'.format(model_name))
    plt.close()


def ablation(dataset, labels, model, total_accuracy, n_segments=4, n_channels=22):

    differences = ablation_zero_segments(dataset, labels, model, total_accuracy, n_segments)
    print("\nAblation with zeros in the segments: \n", differences)

    differences = ablation_linear_segments(dataset, labels, model, total_accuracy, n_segments)
    print("\nAblation with linearity in the segments: \n", differences)

    differences = ablation_zero_channels(dataset, labels, model, total_accuracy, n_channels)
    print("\nAblation with zeros in the channels: \n", differences)


def ablation_zero_segments(dataset, labels, model, accuracy, n_segments=4):  # n.trial * n.channel * n.sample

    differences = np.empty(n_segments)
    length = int(dataset.shape[2] / n_segments)

    for k in range(n_segments):
        data = copy.deepcopy(dataset)

        start = k * length
        end = (k + 1) * length

        if (k + 2) * length > data.shape[2]:
            end = data.shape[2]

        data[:, :, start:end] = np.zeros((data.shape[0], data.shape[1], end - start))

        results = model.evaluate(data, labels, verbose=0)
        differences[k] = accuracy - results[1]

    return differences


def ablation_linear_segments(dataset, labels, model, accuracy, n_segments=4):

    differences = np.empty(n_segments)
    length = int(dataset.shape[2] / n_segments)

    for k in range(n_segments):

        data = copy.deepcopy(dataset)

        start = k * length
        end = (k + 1) * length

        if (k + 2) * length > data.shape[2]:
            end = data.shape[2]

        for j in range(data.shape[0]):
            for i in range(data.shape[1]):

                prev = data[j, i, start]
                cons = data[j, i, end - 1]
                lin_len = end - start

                lin = np.linspace(prev, cons, num=lin_len)
                data[j, i, start:end] = lin

        results = model.evaluate(data, labels, verbose=0)
        differences[k] = accuracy - results[1]

    return differences


def ablation_zero_channels(dataset, labels, model, accuracy, n_channels=22):

    differences = np.empty(n_channels)

    for k in range(n_channels):

        data = copy.deepcopy(dataset)

        data[:, k, :] = np.zeros((data.shape[0], data.shape[2]))
        results = model.evaluate(data, labels, verbose=0)
        differences[k] = accuracy - results[1]

    return differences
