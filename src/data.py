import numpy as np 
import os
import csv

def load_data(data_path):
    """
    Associated test case: tests/test_data.py

    Reading and manipulating data is a vital skill for machine learning.

    This function loads the data in data_path csv into two numpy arrays:
    features (of size NxK) and targets (of size Nx1) where N is the number of rows
    and K is the number of features. 
    
    data_path leads to a csv comma-delimited file with each row corresponding to a 
    different example. Each row contains binary features for each example 
    (e.g. chocolate, fruity, caramel, etc.) The last column indicates the label for the
    example how likely it is to win a head-to-head matchup with another candy 
    bar.

    This function reads in the csv file, and reads each row into two numpy arrays.
    The first array contains the features for each row. For example, in candy-data.csv
    the features are:

    chocolate,fruity,caramel,peanutyalmondy,nougat,crispedricewafer,hard,bar,pluribus

    The second array contains the targets for each row. The targets are in the last 
    column of the csv file (labeled 'class'). The first row of the csv file contains 
    the labels for each column and shouldn't be read into an array.

    Example:
    chocolate,fruity,caramel,peanutyalmondy,nougat,crispedricewafer,hard,bar,pluribus,class
    1,0,1,0,0,1,0,1,0,1

    should be turned into:

    [1,0,1,0,0,1,0,1,0] (features) and [1] (targets).

    This should be done for each row in the csv file, making arrays of size NxK and Nx1.

    Args:
        data_path (str): path to csv file containing the data

    Output:
        features (np.array): numpy array of size NxK containing the K features
        targets (np.array): numpy array of size 1xN containing the N targets.
        attribute_names (list): list of strings containing names of each attribute 
            (headers of csv)
    """

    attribute = open(data_path,'r')
    reader = csv.reader(attribute)
    stuff = next(reader)
    attribute_names = stuff[:-1]
    feat_targ = np.genfromtxt(data_path, delimiter=',', skip_header = 1)
    features = feat_targ[:,:-1]
    targets = feat_targ[:,-1]

    return features, targets, attribute_names


def train_test_split(features, targets, fraction):
    """
    Split features and targets into training and testing, randomly. N points from the data 
    sampled for training and (features.shape[0] - N) points for testing. Where N:

        N = int(features.shape[0] * fraction)
    
    Returns train_features (size NxK), train_targets (Nx1), test_features (size MxK 
    where M is the remaining points in data), and test_targets (Mx1).
    
    Special case: When fraction is 1.0. Training and test splits should be exactly the same. 
    (i.e. Return the entire feature and target arrays for both train and test splits)

    Args:
        features (np.array): numpy array containing features for each example
        targets (np.array): numpy array containing labels corresponding to each example.
        fraction (float between 0.0 and 1.0): fraction of examples to be drawn for training

    Returns
        train_features: subset of features containing N examples to be used for training.
        train_targets: subset of targets corresponding to train_features containing targets.
        test_features: subset of features containing M examples to be used for testing.
        test_targets: subset of targets corresponding to test_features containing targets.
    """
    if (fraction > 1.0):
        raise ValueError('N cannot be bigger than number of examples!')
    N = int(features.shape[0] * fraction)
    M = features.shape[0] - 1

    vertical_targets = np.vstack(targets)
    DATA = np.hstack((features,vertical_targets))  #from campuswire 14
    np.random.permutation(DATA)
    feature = DATA[:,:-1]
    target = DATA[:,-1]
    train_features = feature[:N,:]
    train_targets = target[:N]
    test_features = feature[N:,:]
    test_targets = feature[N:]

    if (fraction == 1.0):
        return features, targets, features, targets
    else:
        return train_features, train_targets, test_features, test_targets


