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

## from k_NN_mnist import l1_distance
## from k_NN_mnist import l2_distance
## from k_NN_mnist import kNN
## Can now access x_train, y_train, x_test, y_test, img_dimensions, n_train, y_train, n_labels

total_training_time = 0
total_training_test_time = 0
total_test_time = 0

training_accuracy = 0
test_accuracy = 0

##
## Measure training time
##
## print("Running k-NN... This will take a while! For this run I have selected k = 245")

k = 100

l1_labels = np.genfromtxt('l1_distance_kNN_1000.csv', delimiter=',')
l2_labels = np.genfromtxt('l2_distance_kNN_1000.csv', delimiter=',')

k_l1_labels = l1_labels[:,0:k]
k_l2_labels = l2_labels[:,0:k]

print(k_l1_labels.shape)
print(k_l2_labels.shape)

l1_predictions = stats.mode(k_l1_labels, axis=1)[0]
l2_predictions = stats.mode(k_l2_labels, axis=1)[0]

start = time.perf_counter()
## prediction = kNN(k, pixel_distance)[0]
end = time.perf_counter()

total_training_time = (end - start)

##
## Test k-NN on test data
##
print("Testing k-NN on test data...")
score = 0

start = time.perf_counter()
##for i in range(mnist.n_test):
##	if (prediction[i] == mnist.y_test[i]):
##		score = score + 1
end = time.perf_counter()
test_accuracy = score / mnist.n_test
total_test_time = (end - start)


## Calculate error
training_error = 1 - training_accuracy
test_error = 1 - test_accuracy

## Report Metrics
print()
print("Total training time (time to create molds) was: " + str(total_training_time) + " seconds")
print("Total test time was (on test data): " + str(total_test_time) + " seconds")
print()
print("Error rate on the test data was: " + str(test_error))
