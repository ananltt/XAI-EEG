from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas


def variability_analysis(output_folder):

    paths = glob(output_folder+'/zero_*.csv')
    paths = paths + glob(output_folder+'/interpolation_*.csv')
    paths = paths + glob(output_folder+'/channel_*.csv')

    for path in paths:
        # Read of the file as dataframe and initialization of the matrix
        df = pandas.read_csv(path, index_col=None, header=None)
        zero_accuracies = np.full((df.shape[0], df.shape[1]), None)

        # Scan of the dataframe rows and save in the data matrix
        for row in range(0, df.shape[0]):
            zero_accuracies[row, :] = np.array(df.iloc[row])

        zero_accuracies = np.array(zero_accuracies, dtype='float64')

        mean = np.mean(zero_accuracies, axis=1)
        std = np.std(zero_accuracies, axis=1)

        name_ext = Path(path).name
        name = name_ext.split('.')[0]

        fig, axs = plt.subplots()
        axs.set_title('Accuracy variability: {}'.format(name))
        axs.boxplot(zero_accuracies)
        plt.tight_layout()
        plt.savefig(output_folder+'/{}'.format(name))
        plt.show()
