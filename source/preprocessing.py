import pandas as pd
from scipy.io import loadmat
from EEGModels import EEGNet


def load_data(reference, index, folder):
    part_id = int(reference.iloc[index][0])
    trial_id = int(reference.iloc[index][1])
    label = str(reference.iloc[index][2])

    if isinstance(folder, bytes):
        folder = folder.decode()
    if isinstance(part_id, bytes):
        part_id = part_id.decode()

    file_name = str(folder + 'S' + str(part_id) + '_' + str(trial_id) + '.mat')
    signal = loadmat(file_name)['s']
    signal.reshape(-1, 22)

    return signal


data_dir = 'dataset/EEG/'
data_file = data_dir + 'reference.csv'
reference = pd.read_csv(data_file, index_col=None, header=0)

index = 0
data = load_data(reference, index, data_dir)
