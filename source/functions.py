from matplotlib import pyplot as plt
from scipy.io import loadmat
import numpy as np
from numpy.fft import rfft
import pywt


def load_dataset(ref, data_dir):
    dataset = []
    labels = []

    for row in range(len(ref)):

        participant_id = int(ref.iloc[int(row)]['subject'])
        trial = int(ref.iloc[int(row)]['trial'])
        left_label = int(ref.iloc[int(row)]['left_label'])
        right_label = int(ref.iloc[int(row)]['right_label'])

        if isinstance(data_dir, bytes):
            data_dir = data_dir.decode()
        if isinstance(participant_id, bytes):
            participant_id = participant_id.decode()

        file_name = str('S' + str(participant_id) + '_' + str(trial))
        file_dat = str(data_dir + '/' + str(file_name) + '.mat')
        data = loadmat(file_dat)['s']

        np.nan_to_num(data, False, 0)  ## IMPORTANTE!!!
        dataset.append(data)
        labels.append([left_label, right_label])

        # for j in range(data.shape[0]):
        #     for k in range(data.shape[1]):
        #         if math.isnan(data[j, k]):
        #             print("Trovato - data {} {} {}".format(file_name, j, k))

    return np.array(dataset), np.array(labels)


def extract_FFT(matrix):
    fft_mat = []

    for trial in matrix:
        trial = np.matrix(trial)
        fft_trial = np.empty((trial.shape[0], 501))

        for i in range(trial.shape[0]):
            data = fft_trial[i, :]
            fft_trial[i, :] = np.abs(np.fft.fft(data))**2

        fft_mat.append(fft_trial)

    return np.array(fft_mat)


def extract_wt(matrix):
    approx_trials = []
    for trial in matrix:
        approx = []
        for channel in trial:
            cA, cD = pywt.dwt(channel, 'db1')
            approx.append(np.concatenate((cA, cD)))
        approx_trials.append(approx)
    return np.array(approx_trials)


def plot_model_training(history):
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


def ablation_zero_segments(dataset, labels, model, accuracy, fs=250):  # dataset: ntrial * nchannel * nsample

    differences = np.empty(4)

    for k in range(4):

        data = dataset

        start = k * fs
        end = (k + 1) * fs

        data[:, :, start:end] = np.zeros((52, 22, end-start))

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
                prev = data[j, i, start-1]
                cons = data[j, i, end-1]
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

