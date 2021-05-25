import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data_dir = 'dataset/EEG/'
data_file = str(data_dir + '/reference.csv')
reference = pd.read_csv(data_file, index_col=None, header=0)
reference.columns = ['subject', 'trial', 'left_label', 'right_label']

# Divide in training (70%), validation (20%) and test (10%)

train_ref, val_ref = train_test_split(reference, test_size=0.2, stratify=reference[['left_label', 'right_label']], random_state=0)
val_ref, test_ref = train_test_split(val_ref, test_size=0.2, stratify=val_ref[['left_label', 'right_label']], random_state=0)

# Common hyperparameters for the subsequent training

batch_size = 32
num_epochs = 2
train_steps = int(np.ceil(len(train_ref)/batch_size))
val_steps = int(np.ceil(len(val_ref)/batch_size))
test_steps = int(np.ceil(len(test_ref)/batch_size))

