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
    print("\nVariability analysis!\n")

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
    print("Baseline left labels: ", baseline_right)

    paths = glob(output_folder + '/zero_*.csv')
    paths = paths + glob(output_folder + '/interpolation_*.csv')
    paths = paths + glob(output_folder + '/channel_*.csv')

    for path in paths:

        name_ext = Path(path).name
        name = name_ext.split('.')[0]
        print('\nAnalysis of:\t', name)

        zero_accuracies = extract_accuracies_matrix(path)

        mean = np.mean(zero_accuracies, axis=1)
        std = np.std(zero_accuracies, axis=1)

        if name.find('left') != -1:
            b = baseline_left
        elif name.find('right') != -1:
            b = baseline_right
        else:
            b = baseline_both

        difference = np.repeat(b, zero_accuracies.shape[0] * zero_accuracies.shape[1]).reshape(zero_accuracies.shape)
        difference = np.subtract(difference, zero_accuracies)
        difference = difference * 100 / b
        mean_difference = np.mean(difference, axis=0)

        print(*mean_difference, sep=" & ")

        if name.find('channel') != -1:
            labels = ['C' + str(i + 1) for i in range(0, zero_accuracies.shape[1])]
        else:
            labels = ['S' + str(i + 1) for i in range(0, zero_accuracies.shape[1])]

        fig, axs = plt.subplots()
        axs.set_title('Accuracy variability: {}'.format(name))
        axs.boxplot(difference, labels=labels)
        axs.set_ylim([-100, 100])
        plt.tight_layout()
        plt.savefig(output_folder + '/{}'.format(name))
        plt.show()
