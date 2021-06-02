from EEGModels import EEGNet
from matplotlib import pyplot as plt

from functions import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from FBCSP import FBCSP

if __name__ == "__main__":

    data_dir = 'dataset/EEG/'
    data_file = str(data_dir + '/reference.csv')
    reference = pd.read_csv(data_file, index_col=None, header=0)
    reference.columns = ['subject', 'trial', 'left_label', 'right_label']

    dataset, labels = load_dataset(reference, data_dir)

    classes = []
    for label in labels:
        if (np.array(label) == np.array([1, 0])).all():
            classes.append(0)
        else:
            classes.append(1)

    values0 = values1 = []
    for i in range(len(labels)):
        if (np.array(labels[i]) == np.array([1, 0])).all():
            values0.append(dataset[i])
        else:
            values1.append(dataset[i])

    dict = {'0': np.stack(values0), '1': np.stack(values1)}
    extractor = FBCSP(dict, fs=250, freqs_band=np.linspace(4, 80, 10))

    # USE OF EEGNET WITHOUT FEATURES!!!!

    # Divide the reference in training (70%), validation (20%) and test (10%)

    train_ref, val_ref = train_test_split(reference, test_size=0.2, stratify=reference[['left_label', 'right_label']],
                                          random_state=0)
    val_ref, test_ref = train_test_split(val_ref, test_size=0.2, stratify=val_ref[['left_label', 'right_label']],
                                         random_state=0)

    # Common hyperparameters for the training

    batch_size = 16
    num_epochs = 30

    # From the reference, create the datasets

    train_dataset, train_labels = load_dataset(train_ref, data_dir)
    train_steps = int(np.ceil(len(train_ref) / batch_size))

    val_dataset, val_labels = load_dataset(val_ref, data_dir)
    val_steps = int(np.ceil(len(val_ref) / batch_size))

    test_dataset, test_labels = load_dataset(test_ref, data_dir)
    test_steps = int(np.ceil(len(test_ref) / batch_size))

    # Data shape

    input_shape = (train_dataset[0].shape[0], train_dataset[0].shape[1], 1)

    # Fitting of EEGNET

    model = EEGNet(nb_classes=2, Chans=input_shape[0], Samples=input_shape[1])
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    history = model.fit(x=train_dataset, y=train_labels, batch_size=batch_size, epochs=num_epochs,
                        validation_data=(val_dataset, val_labels))

    results = model.evaluate(test_dataset, test_labels)
    print("\nTest loss, Test accuracy: ", results)

    # accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('images/model_accuracy.png')
    plt.close()

    # loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('images/model_loss.png')
    plt.close()
