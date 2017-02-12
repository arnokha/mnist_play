##
## Use kNN for mnist
## Not using weighted k-NN. Only going to use the mode of k-NNs or return the first value if no mode exists
##
## Author: Arno Khachatourian
##

import numpy as np
from scipy import stats ## For mode calculation
import extract_mnist_data as mnist ## Now have access to x_train, y_train, x_test, y_test, n_train, n_test, img_dimensions, and n_labels

## TODO
## weighted_k_NN(k, distance_function)
## print_k_NN_with_weighting_to_file(filename, distance_function)

def l2_distance(x1, x2):
	""" This distance function just takes the absolute value of the differences in pixel intensities between these two images """
	return np.sqrt( np.sum( (x1 - x2) ** 2) )

def l1_distance(x1, x2):
	""" This distance function just takes the absolute value of the differences in pixel intensities between these two images """
	return np.sum(abs(x1 - x2))

def print_mnist_kNN_to_file(filename, distance_function):
	""" Since k-NN is so computationally expensive, print labels of all nearest neighbors to a file to read, with k = 1000
	    This will print labels in order of closeness to the test point (closeness determined by given distance function).
	    That way if you want to do, for example, k-NN with k = 9, you just take the mode of the first 9 labels per row.
	"""
	print("Filename is " + filename + " and dfun is " + distance_function.__name__)
	k = 1000
	labels_of_nearest_neighbors = kNN(k, distance_function)[1] ## size is (mnist.n_test, k)
	print("kNN has finished running for k = " + str(k) + " and distance function " + distance_function.__name__)
	print("Printing results to " + str(filename))
	np.savetxt(filename, labels_of_nearest_neighbors, delimiter=",", fmt='%1u')
	print("Done")

## for distance, use 10k-15k
def alt_kNN(distance_threshold, distance_function):
	""" This algorithm is probably already named, but I dont know it, so Im just going to call it alt_kNN.
	    Instead of using k_NN, set a radial distance threshold around a test case and take the mode
	    of those labels as the prediction.
	"""
	## Initialize variables
	#max_distance = 28 * 28 * 255
	d = np.zeros((mnist.n_test, mnist.n_train), dtype=np.uint32) # Max value is 28 * 28 * 255 = 199,920
	prediction = np.zeros((mnist.n_test, 1), dtype=np.uint8) # Max value is 9
	confidence = np.zeros((mnist.n_test, 1))

	##
	## Loop through test set, calculate distance between each test and training example, 
	## find neighbors with difference under threshold, and use the mode of their labels to predict the label of the test example
	##
	for i in range(mnist.n_test):
		label_list = []
		for neighbor in range(mnist.n_train):
			d[i,neighbor] = distance_function( mnist.x_test[i,:], mnist.x_train[neighbor,:] )
			if d[i,neighbor] < distance_threshold:
				label_list.append(mnist.y_train[neighbor])

		if (len(label_list) != 0):
			prediction[i] = stats.mode(label_list)[0][0]
			confidence[i] = stats.mode(label_list)[1][0] / len(label_list)
		else:
			prediction[i] = 10

	return (prediction, confidence)

## TODO optional parameter -- filename with close labels
def kNN(k, distance_function):
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
			d[i,neighbor] = distance_function( mnist.x_test[i,:], mnist.x_train[neighbor,:] )

		if i % 500 == 0:
			print(str(i)) ## For sanity reasons 

		for k_index in range(k):
			index_min_distance = np.argmin(d[i,:])
			d[i,index_min_distance] = max_distance ## Set to max so next iteration wont pick again... need to remove if I'm gonna use weighting
			nearest_neighbors[i,k_index] = index_min_distance
			labels_of_nearest_neighbors[i,k_index] = mnist.y_train[ nearest_neighbors[i,k_index] ]

		predictions[i] = stats.mode(labels_of_nearest_neighbors[i], axis=None)[0][0]
#		print( str(predictions[i]) + "," + str(mnist.y_test[i]) )

	return (predictions, labels_of_nearest_neighbors)

def main():
	k = int(input("Enter a number to use for k-Nearest-Neighbors: "))
	kNN(k, l1_distance)


if __name__ == "__main__":
	main()
