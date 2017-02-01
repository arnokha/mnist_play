##
## Average the cases in the training set to create a "mold" to use for predictions
## We know that this won't work as well as the ML methods, but I want to use this benchmark as the "low bar" to beat
##
## Author: Arno Khachatourian
##

import numpy as np
import extract_mnist_data as mnist
## Can now access x_train, y_train, x_test, y_test, img_dimensions, n_train, y_train, n_labels

def count_labels(labels):
	""" Loop through 1D label array and count the instances of each label. Return count of each label in array form """
	label_count = np.zeros((mnist.n_labels, 1), dtype=np.uint16)

	for i in range(labels.size):
		label_count[ labels[i] ] += 1

	return label_count

def create_molds():
	""" Create molds of numbers by averaging the training examples """
	label_counter = np.zeros( (mnist.n_labels, 1), dtype=np.uint16 )  # Used for mold averaging
	molds = np.zeros( (mnist.n_labels, mnist.img_dimensions), dtype=np.uint32 )

	label_counter = count_labels( mnist.y_train )

	## Create molds (may need to regularize if numbers get too big)
	for i in range( mnist.n_train ):
		label = mnist.y_train[i]
		molds[label,:] = np.add( molds[label,:], mnist.x_train[i,:] )

	## Regularize mold by averaging it by number of cases per label
	for i in range(mnist.n_labels):
		molds[i,:] = molds[i,:] / label_counter[i]

	return molds

def create_regularized_molds():
	""" Create molds of numbers by averaging the training examples, 
	and make sure they have the same overall pixel intensity 
	"""
	r_molds = create_molds() # Start with the non-regularized molds
	mold_pixel_count = np.zeros( (mnist.n_labels, 1), dtype=np.uint16 )
	for label in range(mnist.n_labels):
		for pixel in range(mnist.img_dimensions):
			mold_pixel_count[label] += r_molds[label,pixel]

	# Regularize
	mold_regulaizer = mold_pixel_count / 10000
	for label in range(mnist.n_labels):
		for pixel in range(mnist.img_dimensions):
			r_molds[label,pixel] = r_molds[label,pixel] / mold_regulaizer[label]

	return r_molds

def main():
	## TODO change to pass() create_molds()
	create_regularized_molds()

if __name__ == "__main__":
	main()
