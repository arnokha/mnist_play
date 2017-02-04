##
## Use kNN for mnist
## Not using weighted k-NN. Only going to use the mode of k-NNs or return the first value if no mode exists
##
## Author: Arno Khachatourian
##

import numpy as np
from scipy import stats ## For mode calculation
import extract_mnist_data as mnist ## Now have access to x_train, y_train, x_test, y_test, n_train, n_test, img_dimensions, and n_labels

## Should consider trying out different distance functions to see how the error rate is affected
def distance(x1, x2):
	""" This distance function just takes the absolute value of the differences in pixel intensities between these two images """
	return np.sum(abs(x1 - x2))

def print_mnist_kNN_to_file(filename):
	""" Since k-NN is so computationally expensive, print labels of all nearest neighbors to a file to read, with k = max (60,000)
	    This will print labels in order of closeness to the test point. That way if you want to do, for example, k-NN with k = 9,
	    you just take the mode of the first 9 labels per row.
	"""
	k = 60000
	pass

def kNN(k):
	""" Uses k-NN to return predictions based on the labels of the k nearest neighbors. Does not use weighting """
	## Initialize variables
	max_distance = 28 * 28 * 255
	d = np.zeros((mnist.n_test, mnist.n_train), dtype=np.uint32) # Max value is 28 * 28 * 255 = 199,920
	nearest_neighbors = np.zeros((mnist.n_test, k), dtype=np.uint16) # Max value is 59,999
	labels_of_nearest_neighbors = np.zeros((mnist.n_test, k), dtype=np.uint8) # Max value is 9
	predictions = np.zeros((mnist.n_test, 1), dtype=np.uint8) # Max value is 9

	##
	## Loop through test set, calculate distance between each test and training example, 
	## find k nearest neighbors, and use the mode of their labels to predict the label of the test example
	##
	for i in range(mnist.n_test):
		for neighbor in range(mnist.n_train):
			d[i,neighbor] = distance( mnist.x_test[i,:], mnist.x_train[neighbor,:] )

		if i % 500 == 0:
			print(str(i)) ## For sanity reasons 

		for k_index in range(k):
			index_min_distance = np.argmin(d[i,:])
			d[i,index_min_distance] = max_distance ## Set to max so next iteration wont pick again... need to remove if I'm gonna use weighting
			nearest_neighbors[i,k_index] = index_min_distance
			labels_of_nearest_neighbors[i,k_index] = mnist.y_train[ nearest_neighbors[i,k_index] ]

		predictions[i] = stats.mode(labels_of_nearest_neighbors[i], axis=None)[0][0]
#		print( str(predictions[i]) + "," + str(mnist.y_test[i]) )

	return predictions

def main():

	k = int(input("Enter an odd number to use for k-Nearest-Neighbors: "))
	if ( k % 2 == 0): 
		## Number is even. Prompt user for an odd number
		print(str(k) + " is an even number. Please enter an odd number.")
		exit()
	kNN(k)


if __name__ == "__main__":
	main()