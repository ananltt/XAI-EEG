from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas

if __name__ == "__main__":

    paths = glob('../output/zero_*.csv')
    paths = paths + glob('../output/interpolation_*.csv')
    paths = paths + glob('../output/channel_*.csv')

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

        fig, axs = plt.subplots()
        axs.set_title('Accuracy variability')
        axs.boxplot(zero_accuracies)
        plt.tight_layout()
        plt.show()
