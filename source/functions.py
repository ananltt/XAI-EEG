from scipy.io import loadmat
import numpy as np


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
