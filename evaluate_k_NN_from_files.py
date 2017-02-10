##
## Benchmark the k-NN method of digit classification (with distance function d = sum( abs(x1 - x2) )
## We want to measure:
## - Total time to train (in this case the time to create the molds) (note: this includes extraction time)
## - Total time to test
## - Accuracy on training data
## - Accuracy on test data
##
## Author: Arno Khachatourian 
##

import time
import numpy as np
from scipy import stats ## For mode calculation

print("Extracting MNIST data...")
import extract_mnist_data as mnist
print("Done")

##
## Read labels from preprocessed CSVs
##

l1_labels = np.genfromtxt('knn_results/l1_distance_kNN_1000.csv', delimiter=',')
# L2 sucks for this -- l2_labels = np.genfromtxt('knn_results/l2_distance_kNN_1000.csv', delimiter=',')

##
## For k from 10-1000, (in increments of 10), test the accuracy of k-NN (using both L1 and L2 distances)
##
k_range = range(1,100)

## Big values of k also suck

score_l1 = np.zeros((len(k_range), 1))

i = 0 ## for scoring

for k in k_range:
	## Get k columns of labels
	k_l1_labels = l1_labels[:, 0:k]

	## Predict the mode of the k labels
	l1_predictions = stats.mode(k_l1_labels, axis=1)[0]

	## L1 Scoring
	for test_case in range(mnist.n_test):
		if (l1_predictions[test_case] == mnist.y_test[test_case]):
			score_l1[i] += 1

	i += 1

accuracy_l1 = score_l1 / mnist.n_test 
error_l1 = 1 - accuracy_l1

best_k = np.argmin(error_l1)
smallest_error = np.min(error_l1)

print("The value of k that produced the lowest error rate was " + str(best_k))
print("The accuracy on the test set was " + str(1 - smallest_error))
print("The error on the test set was " + str(smallest_error))
