# -*- coding: utf-8 -*-
"""
File containing various support function.

@author: Anna Nalotto
@credits: Alberto Zancanaro(jesus-333)
@organization: University of Padua (Italy)
"""

#%%
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat



#%%

def loadDatasetD2(path, idx):
    """
    Function to load the dataset 2 of the BCI competition.
    N.B. This dataset is a costum dataset crated from the original gdf file using the MATLAB script 'dataset_transform.m'

    Parameters
    ----------
    path : string
        Path to the folder.
    idx : int.
        Index of the file.

    Returns
    -------
    data : Numpy 2D matrix
        Numpy matrix with the data. Dimensions are "samples x channel".
    event_matrix : Numpy matrix
        Matrix of dimension 3 x number of event. The first row is the position of the event, the second the type of the event and the third its duration

    """
    path_data = path + '/' + str(idx) + '_data.mat' 
    path_event = path + '/' + str(idx) + '_label.mat'
    
    data = loadmat(path_data)['data']
    event_matrix = loadmat(path_event)['event_matrix']
    
    return data, event_matrix
    
    
def computeTrialD2(data, event_matrix, fs, windows_length = 4, remove_corrupt = False):
    """
    Convert the data matrix obtained by loadDatasetD2() into a trials 3D matrix

    Parameters
    ----------
    data : Numpy 2D matrix
        Input data obtained by loadDatasetD2().
    event_matrix : Numpy 2D matrix
        event_matrix obtained by loadDatasetD2().
    fs: int
        frequency sampling
    windows_length: double
        Length of the trials windows in seconds. Defualt is 4.

    Returns
    -------
    trials : Numpy 3D matrix
        Matrix with dimensions "n. trials x channel x n. samples per trial".
    labels : Numpy vector
        Vector with a label for each trials. For more information read http://www.bbci.de/competition/iv/desc_2a.pdf

    """
    event_position = event_matrix[:, 0]
    event_type = event_matrix[:, 1]
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Remove corrupted trials
    if(remove_corrupt):
        event_corrupt_mask_1 = event_type == 1023
        event_corrupt_mask_2 = event_type == 1023
        for i in range(len(event_corrupt_mask_2)):
            if(event_corrupt_mask_2[i] == True): 
                # Start of the trial
                event_corrupt_mask_1[i - 1] = True
                # Type of the trial
                event_corrupt_mask_1[i + 1] = True
                
        
        event_position = event_position[np.logical_not(event_corrupt_mask_1)]
        event_type = event_type[np.logical_not(event_corrupt_mask_1)]
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Since trials have different length I crop them all to the minimum length
    
    # Retrieve event start
    event_start = event_position[event_type == 768]
    
    # Evaluate the samples for the trial window
    start_second = 2
    end_second = 6
    windows_sample = np.linspace(int(start_second * fs), int(end_second * fs) - 1, int(end_second * fs) - int(start_second * fs)).astype(int)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Create the trials matrix
    trials = np.zeros((len(event_start), data.shape[1], len(windows_sample)))
    data = data.T
    
    for i in range(trials.shape[0]):
        trials[i, :, :] = data[:, event_start[i] + windows_sample]
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Create the label vector
    labels = event_type[event_type != 768]
    labels = labels[labels != 32766]
    labels = labels[labels != 1023]
    
    new_labels = np.zeros(labels.shape)
    labels_name = {}
    labels_name[769] = 1
    labels_name[770] = 2
    labels_name[771] = 3
    labels_name[772] = 4
    labels_name[783] = -1
    for i in range(len(labels)):
        new_labels[i] = labels_name[labels[i]]
    labels = new_labels
    
    return trials, labels


def createTrialsDictD2(trials, labels, label_name = None):
    """
    Converts the trials matrix and the labels vector in a dict.

    Parameters
    ----------
    trials : Numpy 2D matrix
        trials matrix obtained by computeTrialD2().
    labels : Numpy vector
        vector trials obtained by computeTrialD2().
    label_name : dicitonary, optional
        If passed must be a dictionart where the keys are the value 769, 770, 771, 772. For each key you must insert the corresponding label.
        See the table 2 at http://www.bbci.de/competition/iv/desc_2a.pdf for  more information.
        The default is None.

    Returns
    -------
    trials_dict : TYPE
        DESCRIPTION.

    """
    trials_dict = {}
    keys = np.unique(labels)
    
    for key in keys:
        if(label_name != None): trials_dict[label_name[key]] = trials[labels == key, :, :]
        else: trials_dict[key] = trials[labels == key, :, :] 
    
    return trials_dict


