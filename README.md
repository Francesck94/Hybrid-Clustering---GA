# Hybrid-Clustering---GA

This repository implements a classifier for a One Class Classification.
The classifier uses a clustering algorithm and a genetic algorithm.

It is structed like this:

Dataset is dived in three disjoint groups: Training Set, Validation Set and Test Set.
The Training Set contains only positive class patterns, while Validation Set and Test Set contains positive and negative class patterns.
A k - means algorithm is applied on Training Set, and for each cluster a radius is computed by the matemathical average of patterns'
distances from centroid.
The k - means algorithm is implemented with a custom weighted distance. Weights are chosen by a genetic algoritmh which targets is maximize
accuracy on validation set.
After the k-means on training set, validation set's patterns are classified as positive or negative according their position: if they are
inside a cluster they labeled as positive, otherwise they are labeled as negative. After classification, accuracy on validation is computed.
The genetic algoritmh creates a new generation of weights, in order to re-make the k-means on training set and classification on validation
set. The process stops when weights from genetic algorithm allow to overcome a threshold in validation set's Accuracy. 
Then test set'patterns are classified.
