# Hybrid-Clustering---GA

This repository implements a classifier for a One Class Classification problem.
The main core of the classifier is a clustering algorithm combined with a genetic algorithm.

It is structed like this:

Dataset is dived in three disjoint groups: Training Set, Validation Set and Test Set.
The Training Set contains only positive class patterns, while Validation Set and Test Set contains both positive and negative class patterns.
A k - means algorithm is applied on Training Set, and for each cluster a radius is computed by the matemathical average of patterns'
distances from centroid.
The k - means algorithm is implemented with a custom weighted distance. Weights are chosen by a genetic algoritmh which targets is to maximize accuracy on validation set.
After the k-means on training set, validation set's patterns are classified as positive or negative according their position: if they are
inside a cluster they labeled as positive, otherwise they are labeled as negative. After classification, accuracy on validation is computed.
The genetic algoritmh creates a new generation of weights, in order to re-make the k-means on training set and classification on validation
set. The process stops when weights from genetic algorithm allow to overcome a threshold in validation set's Accuracy. 
Then test set'patterns are classified.

In addition a soft classification is realized, that is for each test set pattern a score, which quantifies his distance from the centroid, is computed. This score comes from a gaussian membership functions and its range is [0,1].

## Calibration ##
In order to obtain posterior probability for test pattern membership, a post - processing calibration phase has been added to the project. The calibration folder contains some methods able to transform score values into probabilities values (Platt scaling, Isotonic Regression, SpineCalib). A calibration function is learned on a calibration set composed by target label y and score values s, and then it is applied to test set score in order to obtain probabilities.
