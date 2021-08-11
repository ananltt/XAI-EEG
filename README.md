# XAI-EEG

## Authors
[Giulia Pezzutti](https://github.com/giuliapezzutti) - [Anna Nalotto](https://github.com/annanltt)

University of Padua, Italy

## Project execution

1. Load the 2a dataset (BCI competition) inside the 'dataset' folder.

2. Execute **dataloading.m** script: it requires the 2a dataset (of BCI competition) downloaded in 'dataset' folder; it extracts data and the correspondent labels in 'dataset/EEG' folder. To modify the path to the folder, it is necessary to change it at line 9 of the cited file. 

3. Execute **main.py**: it loads properly the dataset and divides it into training, validation and test sets. It firstly trains the EEGNet and applies the ablation to the signal dataset, then applies the same procedure to wavelet and FBCSP datasets. 

Finally, in test.py are reported some other networks tested for this project. 

## Project specification

RESEARCH PURPOSE: To classify movements via EEG, and to explain what pieces of the input drive the model's classification decision.

DATASET: BCI competition IV, dataset 2a (choose 2 classes, e.g., right/left hand, or right hand/feet)

TASKs:

(1) Extract features from the dataset (e.g., via filter bank common spatial patterns - FBCSP), and represent each EEG trial with a feature vector.

(2) Choose one possible classification method (e.g., EEGNet based on Python) and classify the dataset (2-class).

(3) Choose one method for checking explainability (e.g., global explainability with ablation) and apply that on the dataset.

## Project structure

All the source code is reported in 'source' folder. All the implemented functions can be found in the 'source' folder. In particular, functions_dataset.py contains all the functions needed for the dataset creation and the feature extractions, functions_networks.py contains the functions for the network training, ablation and permutation mechanisms. FBCSP.py (https://github.com/jesus-333/FBCSP-Python) and EEGNet (https://github.com/vlawhern/arl-eegmodels) are classes already available on GitHub but here reported for the sake of convenience of the user. 

All the output will be saved in 'output' folder, while the models and the plots regarding the training procedures will be saved in the 'models' folder. 

### Requirements

- Biosig toolbox in Matlab
- tensorflow package in Python
