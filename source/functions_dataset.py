import scipy
from scipy import signal
from scipy.stats import skew, kurtosis, entropy
from sklearn.preprocessing import normalize
from scipy.io import loadmat
import numpy as np
from numpy.fft import rfft
import pywt

from source.FBCSP_V4 import FBCSP_V4


def load_dataset(data_dir, subject, fs=250, start_second=2, signal_length=4, consider_artefacts=True):

    i = subject

    path_data = data_dir + '/S' + str(i) + '_data.mat'
    path_event = data_dir + '/S' + str(i) + '_label.mat'

    data = loadmat(path_data)['data']
    event_matrix = loadmat(path_event)['event_matrix']

    event_position = event_matrix[:, 0]
    event_type = event_matrix[:, 1]

    positions = []
    labels = []

    start_types = [768, 1023 if consider_artefacts is True else None]

    for l in range(len(event_type)):
        if (event_type[l] in start_types) and event_type[l + 1] == 769:
            positions.append(l)
            labels.append(769)

        if (event_type[l] in start_types) and event_type[l + 1] == 770:
            positions.append(l)
            labels.append(770)

    event_start = event_position[positions]

    # Evaluate the samples for the trial window
    end_second = start_second + signal_length
    windows_sample = np.linspace(int(start_second * fs), int(end_second * fs) - 1,
                                 int(end_second * fs) - int(start_second * fs)).astype(int)

    # Create the trials matrix
    trials = np.zeros((len(event_start), data.shape[1], len(windows_sample)))
    data = data.T

    for j in range(trials.shape[0]):
        trials[j, :, :] = normalize(data[:, event_start[j] + windows_sample], axis=1)

    new_labels = []
    labels_name = {769: [1, 0], 770: [0, 1]}
    for j in range(len(labels)):
        new_labels.append(labels_name[labels[j]])
    labels = new_labels

    return trials, labels


def create_dict(dataset, labels, labels_name):

    values1 = []
    values2 = []

    for i in range(len(labels)):
        if (labels[i] == [1, 0]).all():
            values1.append(dataset[i])
        else:
            values2.append(dataset[i])

    data_dict = {labels_name[1]: np.stack(values1), labels_name[2]: np.stack(values2)}

    return data_dict


def extract_indexes_segments(data_length, n_segments):

    segment_length = int(data_length / n_segments)
    indexes = []

    for k in range(n_segments):

        start = k * segment_length
        end = (k + 1) * segment_length

        if (k + 2) * segment_length > data_length:
            end = data_length

        indexes.append((start, end))

    return indexes


def extract_FFT(matrix, fs=250):
    fft_dataset = np.zeros((matrix.shape[0], matrix.shape[1], 501))

    for t, trial in enumerate(matrix):

        trial = np.matrix(trial)

        for i in range(trial.shape[0]):
            data = trial[i, :]
            freqs, psd = signal.periodogram(data, fs=fs)
            fft_dataset[t, i, :] = psd

    return np.array(fft_dataset)


def extract_wt(matrix):
    approx_trials = []
    for trial in matrix:
        approx = []
        for channel in trial:
            cA, cD = pywt.dwt(channel, 'db1')
            approx.append(np.concatenate((cA, cD)))
        approx_trials.append(approx)
    return np.array(approx_trials)


def extract_statistical_characteristics(matrix):

    sc_dataset = np.zeros((matrix.shape[0], matrix.shape[1], matrix.shape[1] + 10))

    for t, trial in enumerate(matrix):
        trial = np.matrix(trial)

        for i in range(trial.shape[0]):
            data = trial[i, :]
            sc = scipy.stats.describe(data, axis=1)
            sc_dataset[t, i, 0] = sc.mean
            sc_dataset[t, i, 1] = np.mean(np.square(data))
            sc_dataset[t, i, 2] = sc.variance
            sc_dataset[t, i, 3] = sc.skewness
            sc_dataset[t, i, 4] = sc.kurtosis
            sc_dataset[t, i, 5] = entropy(data, axis=1)
            sc_dataset[t, i, 6] = np.trapz(np.array(data), axis=1)
            sc_dataset[t, i, 7] = len(data) - np.count_nonzero(data)
            sc_dataset[t, i, 8] = np.max(data) - np.min(data)
            sc_dataset[t, i, 9] = 0 #TODO: trovare un'altra caratteristica statistica

        sc_dataset[t, :, 10:] = np.corrcoef(trial)

    return np.array(sc_dataset)


def extract_pearson_coefficients(matrix):

    coeffs = np.zeros((matrix.shape[0], matrix.shape[1], matrix.shape[1]))

    for t, trial in enumerate(matrix):
        trial = np.matrix(trial)
        coeffs[t, :, :] = np.corrcoef(trial)

    return np.array(coeffs)


def extractFBCSP(data, labels, n_features, fs=250):

    labels_name = {769: 'left', 770: 'right', 1: 'left', 2: 'right'}

    trials_dict = create_dict(data, labels, labels_name)

    FBCSP_f = FBCSP_V4(trials_dict, fs, n_w=2, n_features=n_features, print_var=True)

    fbcsp = FBCSP_f.extractFeaturesForTraining()
    fbcsp = np.concatenate((fbcsp[0], fbcsp[1]), axis=0)

    fbcsp = fbcsp.reshape((fbcsp.shape[0], 1, fbcsp.shape[1]))

    return np.array(fbcsp)
