import numpy as np

from EEGModels import EEGNet
from matplotlib import pyplot as plt


def training_EEGNet(train_data, train_labels, val_data, val_labels, batch_size, num_epochs, model_name):

    input_shape = (train_data[0].shape[0], train_data[0].shape[1], 1)

    model = EEGNet(nb_classes=2, Chans=input_shape[0], Samples=input_shape[1])
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    history = model.fit(x=train_data, y=train_labels, batch_size=batch_size, epochs=num_epochs,
                        validation_data=(val_data, val_labels))
    plot_model_training(history)
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


def ablation(dataset, labels, model, total_accuracy, fs=250):

    differences = ablation_zero_segments(dataset, labels, model, total_accuracy)
    print("\nAblation with zeros in the segments: ", differences)

    differences = ablation_linear_segments(dataset, labels, model, total_accuracy)
    print("\nAblation with linearity in the segments: ", differences)

    differences = ablation_zero_channels(dataset, labels, model, total_accuracy)
    print("\nAblation with zeros in the channels: ", differences)


def ablation_zero_segments(dataset, labels, model, accuracy, fs=250):  # dataset: ntrial * nchannel * nsample

    differences = np.empty(4)

    for k in range(4):
        data = dataset

        start = k * fs
        end = (k + 1) * fs

        data[:, :, start:end] = np.zeros((52, 22, end - start))

        results = model.evaluate(data, labels)
        differences[k] = accuracy - results[1]

    return differences


def ablation_linear_segments(dataset, labels, model, accuracy, fs=250):
    differences = np.empty(4)

    for k in range(4):

        data = dataset

        start = k * fs
        end = (k + 1) * fs

        for j in range(data.shape[0]):
            for i in range(data.shape[1]):
                prev = data[j, i, start - 1]
                cons = data[j, i, end - 1]
                lin = np.linspace(prev, cons, num=250)
                data[j, i, start:end] = lin

        results = model.evaluate(data, labels)
        differences[k] = accuracy - results[1]

    return differences


def ablation_zero_channels(dataset, labels, model, accuracy):
    differences = np.empty(22)

    for k in range(22):
        data = dataset

        data[:, k, :] = np.zeros((52, 1000))
        results = model.evaluate(data, labels)
        differences[k] = accuracy - results[1]

    return differences
