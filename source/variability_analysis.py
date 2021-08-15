from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas


def extract_accuracies_matrix(path):
    # Read of the file as dataframe and initialization of the matrix
    df = pandas.read_csv(path, index_col=None, header=None)
    accuracies = np.full((df.shape[0], df.shape[1]), None)

    # Scan of the dataframe rows and save in the data matrix
    for row in range(0, df.shape[0]):
        accuracies[row, :] = np.array(df.iloc[row])

    return np.array(accuracies, dtype='float64')


def variability_analysis(output_folder):
    print("\nVariability analysis ...\n")

    path = output_folder + '/tot_accuracies.csv'
    accuracies = extract_accuracies_matrix(path)[0]
    baseline_both = np.mean(accuracies, axis=0)
    print("Baseline both labels: ", baseline_both)

    path = output_folder + '/tot_left_accuracies.csv'
    accuracies = extract_accuracies_matrix(path)[0]
    baseline_left = np.mean(accuracies, axis=0)
    print("Baseline left labels: ", baseline_left)

    path = output_folder + '/tot_right_accuracies.csv'
    accuracies = extract_accuracies_matrix(path)[0]
    baseline_right = np.mean(accuracies, axis=0)
    print("Baseline right labels: ", baseline_right)

    paths = glob(output_folder + '/zero_*.csv')
    paths = paths + glob(output_folder + '/interpolation_*.csv')
    paths = paths + glob(output_folder + '/channel_*.csv')
    paths = paths + glob(output_folder + '/permutation_*.csv')

    for path in paths:

        name_ext = Path(path).name
        name = name_ext.split('.')[0]
        print('\nAnalysis of:\t', name)

        zero_accuracies = extract_accuracies_matrix(path)

        mean = np.mean(zero_accuracies, axis=1)
        std = np.std(zero_accuracies, axis=1)

        if name.find('left') != -1:
            b = baseline_left
            title = 'Test Left Hand - '
        elif name.find('right') != -1:
            b = baseline_right
            title = 'Test Right Hand - '
        else:
            b = baseline_both
            title = 'Test Both Hands - '

        difference = np.repeat(b, zero_accuracies.shape[0] * zero_accuracies.shape[1]).reshape(zero_accuracies.shape)
        difference = np.subtract(difference, zero_accuracies)
        difference = difference * 100 / b
        mean_difference = np.mean(difference, axis=0)
        for i, val in enumerate(mean_difference):
            mean_difference[i] = round(val, 3)

        print(*mean_difference, sep="\% & ")

        dict_channels = {'C1': 'FZ', 'C2': 'FC3', 'C3': 'FC1', 'C4': 'FCz', 'C5': 'FC2',
                         'C6': 'FC4', 'C7': 'C5', 'C8': 'C3', 'C9': 'C1', 'C10': 'CZ', 'C11': 'C2', 'C12': 'C4',
                         'C13': 'C6', 'C14': 'CP3', 'C15': 'CP1', 'C16': 'CPz', 'C17': 'CP2', 'C18': 'CP4',
                         'C19': 'P1', 'C20': 'PZ', 'C21': 'P2', 'C22': 'POz'}

        if name.find('channel') != -1:
            labels = ['C' + str(i + 1) for i in range(0, zero_accuracies.shape[1])]
            labels = [dict_channels.get(l) for l in labels]
            title = title + 'Channel '
        else:
            labels = ['s' + str(i + 1) for i in range(0, zero_accuracies.shape[1])]
            title = title + 'Segment '

        if name.find('permutation') != -1:
            title = title + 'Permutation'
        elif name.find('zero') != -1:
            title = title + 'Zero-Ablation'
        else:
            title = title + 'Interpolation-Ablation'

        fig, axs = plt.subplots(figsize=(12, 8))
        axs.set_title(title)
        axs.boxplot(difference, labels=labels)
        axs.set_ylim([-50, 100])
        plt.tight_layout()
        plt.savefig(output_folder + '/{}'.format(name))
        plt.show()


def print_channel_results(output_folder):
    
    path = output_folder + '/tot_accuracies.csv'
    accuracies = extract_accuracies_matrix(path)[0]
    baseline_both = np.mean(accuracies, axis=0)

    path = output_folder + '/tot_left_accuracies.csv'
    accuracies = extract_accuracies_matrix(path)[0]
    baseline_left = np.mean(accuracies, axis=0)

    path = output_folder + '/tot_right_accuracies.csv'
    accuracies = extract_accuracies_matrix(path)[0]
    baseline_right = np.mean(accuracies, axis=0)
    
    path1 = output_folder + '/permutation_channel_accuracies.csv'
    path2 = output_folder + '/permutation_channel_left_accuracies.csv'
    path3 = output_folder + '/permutation_channel_right_accuracies.csv'
    
    col1 = extract_accuracies_matrix(path1)
    col2 = extract_accuracies_matrix(path2)
    col3 = extract_accuracies_matrix(path3)

    difference = np.repeat(baseline_both, col1.shape[0] * col1.shape[1]).reshape(col1.shape)
    difference = np.subtract(difference, col1)
    difference = difference * 100 / baseline_both
    mean_difference1 = np.mean(difference, axis=0)
    for i, val in enumerate(mean_difference1):
        mean_difference1[i] = round(val, 3)
    difference = np.repeat(baseline_left, col2.shape[0] * col2.shape[1]).reshape(col2.shape)
    difference = np.subtract(difference, col2)
    difference = difference * 100 / baseline_left
    mean_difference2 = np.mean(difference, axis=0)
    for i, val in enumerate(mean_difference2):
        mean_difference2[i] = round(val, 3)
    difference = np.repeat(baseline_right, col3.shape[0] * col3.shape[1]).reshape(col3.shape)
    difference = np.subtract(difference, col3)
    difference = difference * 100 / baseline_right
    mean_difference3 = np.mean(difference, axis=0)
    for i, val in enumerate(mean_difference3):
        mean_difference3[i] = round(val, 3)

    dict_channels = {'C1': 'FZ', 'C2': 'FC3', 'C3': 'FC1', 'C4': 'FCz', 'C5': 'FC2',
                         'C6': 'FC4', 'C7': 'C5', 'C8': 'C3', 'C9': 'C1', 'C10': 'CZ', 'C11': 'C2', 'C12': 'C4',
                         'C13': 'C6', 'C14': 'CP3', 'C15': 'CP1', 'C16': 'CPz', 'C17': 'CP2', 'C18': 'CP4',
                         'C19': 'P1', 'C20': 'PZ', 'C21': 'P2', 'C22': 'POz'}

    labels = ['C' + str(i + 1) for i in range(0, col1.shape[1])]
    labels = [dict_channels.get(l) for l in labels]

    for i in range(len(mean_difference1)):
        print(labels[i], ' & ', mean_difference1[i], '\% & ', mean_difference2[i], '\% & ', mean_difference3[i], '\% \\\\')
        print('\hline')
