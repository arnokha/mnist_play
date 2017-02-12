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

print("Extracting MNIST data...")
import extract_mnist_data as mnist
print("Done")

from k_NN_mnist import l1_distance
from k_NN_mnist import alt_kNN
## Can now access x_train, y_train, x_test, y_test, img_dimensions, n_train, y_train, n_labels

total_training_time = 0
total_training_test_time = 0
total_test_time = 0

training_accuracy = 0
test_accuracy = 0

##
## Measure training time
##
print("Running k-NN... This will take a while! For this run I have selected k = 245")
distance_threshold = 17000

start = time.perf_counter()
(prediction, confidence) = alt_kNN(distance_threshold, l1_distance)
end = time.perf_counter()

total_training_time = (end - start)

##
## Test k-NN on test data
##
print("Testing k-NN on test data...")
score = 0
unknowns = 0
errors_on_guess = 0

start = time.perf_counter()
for i in range(mnist.n_test):
	if (prediction[i] == mnist.y_test[i]):
		score = score + 1
	elif (prediction[i] == 10):
		unknowns += 1
	else:
		print("Error on guess. Predicted " + str(prediction[i]) + " with confidence " + str(confidence[i]) + ". Actual label was " + str(mnist.y_test[i]) )
		errors_on_guess += 1

end = time.perf_counter()
test_accuracy = score / mnist.n_test
test_accuracy_on_guess = score / (mnist.n_test - unknowns)
total_test_time = (end - start)


## Calculate error
training_error = 1 - training_accuracy
test_error = 1 - test_accuracy

## Report Metrics
print()
print("Total training time was: " + str(total_training_time) + " seconds")
print("Total test time was : " + str(total_test_time) + " seconds")
print()
print("Error rate on the test data was: " + str(test_error))
print("Accuracy on test data was: " + str(test_accuracy))
print("Accuracy on test data on guess was: " + str(test_accuracy_on_guess))
print("Number of unknowns was " + str(unknowns))
print("Number of errors on guess was: " + str(errors_on_guess))
