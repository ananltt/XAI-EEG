# XAI-EEG

RESEARCH PURPOSE: To classify movements via EEG, and to explain what pieces of the input drive the model's classification decision.

DATASET: BCI competition IV, dataset 2a (choose 2 classes, e.g., right/left hand, or right hand/feet)



METHODS:

- DL-based EEGNet (Python implementation available) or another DL/ML classifier method

- explainability tools (i.e., global/local, ablation/permutation/...)



TASKs:

(1) Extract features from the dataset (e.g., via filter bank common spatial patterns - FBCSP), and represent each EEG trial with a feature vector.

(2) Choose one possible classification method (e.g., EEGNet based on Python) and classify the dataset (2-class).

(3) Choose one method for checking explainability (e.g., global explainability with ablation) and apply that on the dataset.
