from EEGModels import EEGNet
from functions import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    data_dir = 'dataset/EEG/'
    data_file = str(data_dir + '/reference.csv')
    reference = pd.read_csv(data_file, index_col=None, header=0)
    reference.columns = ['subject', 'trial', 'left_label', 'right_label']

    # Divide the reference in training (70%), validation (20%) and test (10%)

    train_ref, val_ref = train_test_split(reference, test_size=0.2, stratify=reference[['left_label', 'right_label']],
                                          random_state=0)
    val_ref, test_ref = train_test_split(val_ref, test_size=0.2, stratify=val_ref[['left_label', 'right_label']],
                                         random_state=0)

    # Common hyperparameters for the training

    batch_size = 16
    num_epochs = 10

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
    print("Test loss, Test accuracy: ", results)
