from functions_dataset import *
from functions_network import *
from variability_analysis import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

if __name__ == "__main__":

    data_folder = '../dataset/EEG'
    output_folder = '../output/variability - 1000 iterations - Wavelet CNN'

    n_segments = 8          # number of segments considered in the signal
    iterations = 1000       # number of iterations of the training for the variability analysis
    eegnet = False          # if perform eegnet training or cnn training
    wavelet = False          # if use the wavelet transform or not

    necessary_redimension = False
    fs = 250                # sampling frequency

    # Accuracies obtained with the entire test set
    tot_accuracies, tot_left_accuracies, tot_right_accuracies = [], [], []
    # Accuracies obtained with zero-ablation
    zero_accuracies, zero_left_accuracies, zero_right_accuracies = [], [], []
    # Accuracies obtained with interpolation-ablation
    interpolation_accuracies, interpolation_left_accuracies, interpolation_right_accuracies = [], [], []
    # Accuracies obtained with channel-ablation
    channel_accuracies, channel_left_accuracies, channel_right_accuracies = [], [], []
    # Accuracies obtained with permutation
    accuracies_permutation, left_accuracies_permutation, right_accuracies_permutation = [], [], []
    # Accuracies obtained with channel-permutation
    channel_accuracies_permutation, channel_left_accuracies_permutation, channel_right_accuracies_permutation = [], [], []

    # sys.stdout = open("../output/output - {} segments.txt".format(n_segments), "w")  # TO WRITE ALL OUTPUT IN A FILE

    subjects = range(1, 10, 1)  # dataset composition
    dataset, labels = None, None

    # For each subject, extract the dataset
    for subject in subjects:

        if dataset is None:
            dataset, labels = load_dataset(data_folder, subject, consider_artefacts=False)

        else:
            d, l = load_dataset(data_folder, subject, consider_artefacts=False)
            dataset = np.concatenate((dataset, np.array(d)), axis=0)  # complete dataset
            labels = np.concatenate((labels, np.array(l)), axis=0)

    labels = np.array(labels)

    # Common hyperparameters for the training
    batch_size = 16
    num_epochs = 50

    # Repeat training and analysis for every iteration
    for i in range(iterations):
        print('\n\tIteration: ', i)

        # Extraction of the different datasets
        train_dataset, test_dataset, train_labels, test_labels = train_test_split(dataset, labels, train_size=0.8)

        wavelet_variation(train_dataset[0][0])
        permutation_visualization(train_dataset[0][0], train_dataset[1][0])

        if wavelet:
            train_dataset_proc = extract_wt(train_dataset)
            test_dataset_proc = extract_wt(test_dataset)
        else:
            train_dataset_proc = train_dataset
            test_dataset_proc = test_dataset

        # Network training
        if eegnet:
            model = training_EEGNet(train_dataset_proc, train_labels, batch_size=batch_size, num_epochs=num_epochs,
                                    model_path='../models/model', necessary_redimension=necessary_redimension)
        else:
            model = training_CNN(train_dataset_proc, train_labels, batch_size=batch_size, num_epochs=num_epochs,
                                 model_path='../models/model', necessary_redimension=necessary_redimension)

        # Network evaluation
        if necessary_redimension:
            test_dataset_proc = np.expand_dims(test_dataset_proc, 3)

        results = model.evaluate(test_dataset_proc, test_labels, verbose=0)
        tot_accuracies.append(results[1])

        # Ablation application
        if wavelet:
            function = extract_wt
        else:
            function = None

        accuracies = ablation(test_dataset, test_labels, model, function, n_segments, necessary_redimension=necessary_redimension)
        zero_accuracies.append(list(accuracies[0]))
        interpolation_accuracies.append(list(accuracies[1]))
        channel_accuracies.append(list(accuracies[2]))

        # Permutation application
        accuracies = permutation(test_dataset, test_labels, model, function, n_segments, necessary_redimension=necessary_redimension)
        accuracies_permutation.append(list(accuracies[0]))
        channel_accuracies_permutation.append(list(accuracies[1]))

        # Division of the dataset according to unique labels
        classes, indexes = np.unique(test_labels, return_inverse=True, axis=0)

        for j, c in enumerate(classes):

            print("Considering labels {}".format(c))

            # Build the dataset corresponding to each label, in case applying the feature extractor algorithm
            data = np.array([test_dataset[i] for i in range(len(indexes)) if indexes[i] == j])

            if wavelet:
                x = extract_wt(data)
            else:
                x = data

            lab = np.repeat([c], data.shape[0], axis=0)

            # Evaluate the model with the built dataset with ablation and permutation
            if necessary_redimension:
                x = np.expand_dims(x, 3)
            results = model.evaluate(x, lab, verbose=0)

            accuracies_ab = ablation(data, lab, model, function, n_segments, necessary_redimension=necessary_redimension)
            accuracies_pe = permutation(data, lab, model, function, n_segments, necessary_redimension=necessary_redimension)

            # Save the results according to the label
            if all((c == [1, 0])):
                tot_left_accuracies.append(results[1])
                zero_left_accuracies.append(list(accuracies_ab[0]))
                interpolation_left_accuracies.append(list(accuracies_ab[1]))
                channel_left_accuracies.append(list(accuracies_ab[2]))
                left_accuracies_permutation.append(list(accuracies_pe[0]))
                channel_left_accuracies_permutation.append(list(accuracies_pe[1]))
            else:
                tot_right_accuracies.append(results[1])
                zero_right_accuracies.append(list(accuracies_ab[0]))
                interpolation_right_accuracies.append(list(accuracies_ab[1]))
                channel_right_accuracies.append(list(accuracies_ab[2]))
                right_accuracies_permutation.append(list(accuracies_pe[0]))
                channel_right_accuracies_permutation.append(list(accuracies_pe[1]))

    # Save output in different files
    save(tot_accuracies, output_folder+"/tot_accuracies.csv")
    save(zero_accuracies, output_folder+"/zero_accuracies.csv")
    save(interpolation_accuracies, output_folder+"/interpolation_accuracies.csv")
    save(channel_accuracies, output_folder+"/channel_accuracies.csv")

    save(tot_left_accuracies, output_folder+"/tot_left_accuracies.csv")
    save(zero_left_accuracies, output_folder+"/zero_left_accuracies.csv")
    save(interpolation_left_accuracies, output_folder+"/interpolation_left_accuracies.csv")
    save(channel_left_accuracies, output_folder+"/channel_left_accuracies.csv")

    save(tot_right_accuracies, output_folder+"/tot_right_accuracies.csv")
    save(zero_right_accuracies, output_folder+"/zero_right_accuracies.csv")
    save(interpolation_right_accuracies, output_folder+"/interpolation_right_accuracies.csv")
    save(channel_right_accuracies, output_folder+"/channel_right_accuracies.csv")

    save(accuracies_permutation, output_folder+"/permutation_accuracies.csv")
    save(channel_accuracies_permutation, output_folder+"/permutation_channel_accuracies.csv")
    save(left_accuracies_permutation, output_folder+"/permutation_left_accuracies.csv")
    save(channel_left_accuracies_permutation, output_folder+"/permutation_channel_left_accuracies.csv")
    save(right_accuracies_permutation, output_folder+"/permutation_right_accuracies.csv")
    save(channel_right_accuracies_permutation, output_folder+"/permutation_channel_right_accuracies.csv")

    # Apply variability analysis
    box_plot_tot_accuracies(output_folder=output_folder)
    variability_analysis(output_folder=output_folder)
    print_channel_results(output_folder=output_folder)