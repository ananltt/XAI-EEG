# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:01:23 2021

@author: Anna Nalotto
"""

# cleanWorkspaec()

#%%
from FBCSP_support_function import loadDatasetD2, computeTrialD2, createTrialsDictD2
import FBCSP

fs = 250
n_w = 2
n_features = 4

labels_name = {}
labels_name[769] = 'left'
labels_name[770] = 'right'
labels_name[771] = 'foot'
labels_name[772] = 'tongue'
labels_name[783] = 'unknown'
labels_name[1] = 'left'
labels_name[2] = 'right'
labels_name[3] = 'foot'
labels_name[4] = 'tongue'

print_var = True



idx_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# idx_list = [1, 2, 3, 6, 7, 8]
# idx_list = [4]

repetition = 5

#%%


for idx in idx_list:
    # for idx in range(1, 10):
        print('Subject n.', str(idx))
        
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Train
        
        # Path for 4 classes dataset
        path_train = 'mat'
        #path_train_label = 'Dataset/D2/v1/True Label/A0' + str(idx) + 'T.mat'
        
        data, event_matrix = loadDatasetD2(path_train, idx)
        
        trials, labels = computeTrialD2(data, event_matrix, fs, remove_corrupt = False)
        #trials = [288,22,1000] element of the dict = [72,22,1000]
        
        
        
        trials_dict = createTrialsDictD2(trials, labels, labels_name)        

#%%
# we want a dictionary with 2 classes (right and left)
[trials_dict.pop(key) for key in ['foot', 'tongue']] #removes the 2 classes that we don't care
#%%

FBCSP_f = FBCSP(trials_dict, fs, n_w = 2, n_features = n_features, print_var = True)

#%%
data_matrix, label = FBCSP_f.createDataMatrix()

#data_matrix = 144 x 8
#n. features = 8
#label = 144