# XAI-EEG

## Authors
[Giulia Pezzutti](https://github.com/giuliapezzutti) - [Anna Nalotto](https://github.com/annanltt)

University of Padua, Italy

## Project specification

RESEARCH PURPOSE: To classify movements via EEG, and to explain what pieces of the input drive the model's classification decision.

DATASET: BCI competition IV, dataset 2a (choose 2 classes, e.g., right/left hand, or right hand/feet)

TASKs:

(1) Extract features from the dataset (e.g., via filter bank common spatial patterns - FBCSP), and represent each EEG trial with a feature vector.

(2) Choose one possible classification method (e.g., EEGNet based on Python) and classify the dataset (2-class).

(3) Choose one method for checking explainability (e.g., global explainability with ablation) and apply that on the dataset.

## Project structure

**NOTE:**
if line 12 of main.py is not commented, all the standard output will be saved in the text file 'output/output.txt' and not shown with the standard visualization. 

All the source code is reported in 'source' folder.

As first step, it is necessary to execute dataloading.m script: it needs to find the dataset download in 'dataset' folder, and it extracts the data, and the correspondent labels in 'dataset/EEG' folder. To modify the path to the folder, it is necessary to change it at line 9 of the cited file. 

As second step, it is necessary to execute main.py: it loads properly the dataset and divide it in training, validation and test set. It firstly applies the EEGNet and the subsequent ablation to the signal dataset and then applies them to wavelet and FBCSP datasets. 

In particular, functions_dataset.py contains all the functions needed for the dataset creation and the feature extractions, functions_networks.py contains the functions for the network training, ablation and permutation mechanisms. FBCSP.py (https://github.com/jesus-333/FBCSP-Python) and EEGNet (https://github.com/vlawhern/arl-eegmodels) are classes already available on GitHub but here reported for the sake of convenience of the user. 

Finally, in 'other_scripts/test.py' are reported some other networks tested for this project. 

All the output will be saved in 'output' folder, while the models will be saved in the 'models' folder. 
