##
## Benchmark the "mold matching" method of digit classification
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

from create_digit_molds import create_molds
## Can now access x_train, y_train, x_test, y_test, img_dimensions, n_train, y_train, n_labels

total_training_time = 0
total_training_test_time = 0
total_test_time = 0

training_accuracy = 0
test_accuracy = 0

## Measure training time
print("Creating molds...")
start = time.perf_counter()
molds = create_molds()
end = time.perf_counter()

total_training_time = (end - start)

##
## Test molds on training data
##
print("Testing molds on training data...")
score = 0
label_score = np.zeros((mnist.n_labels, 1))

start = time.perf_counter()
for i in range(mnist.n_train):
	for j in range(mnist.n_labels):
		label_score[j] = np.sum( np.multiply(mnist.x_train[i,:], molds[j,:]) ) 
	prediction = np.argmax(label_score)
	if (prediction == mnist.y_train[i]):
		score = score + 1

end = time.perf_counter()
training_accuracy = score / mnist.n_train
total_training_test_time = (end - start)

##
## Test molds on test data
##
print("Testing molds on test data...")
score = 0

start = time.perf_counter()
for i in range(mnist.n_test):
	for j in range(mnist.n_labels):
		label_score[j] = np.sum( np.multiply(mnist.x_test[i,:], molds[j,:]) ) 
	prediction = np.argmax(label_score)
	if (prediction == mnist.y_test[i]):
		score = score + 1

end = time.perf_counter()
test_accuracy = score / mnist.n_test
total_test_time = (end - start)


## Calculate error
training_error = 1 - training_accuracy
test_error = 1 - test_accuracy

## Report Metrics
print()
print("Total training time (time to create molds) was: " + str(total_training_time) + " seconds")
print("Total test time was (on training data): " + str(total_training_test_time) + " seconds")
print("Total test time was (on test data): " + str(total_test_time) + " seconds")
print()
print("Error rate on the training data was: " + str(training_error))
print("Error rate on the test data was: " + str(test_error))
