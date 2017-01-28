##
## Print a few examples of the training and test data in human readable form
##
## Author: Arno Khachatourian
##

import numpy as np
import extract_mnist_data as mnist 
## Can now use variables x_train, y_train, x_test, y_test, img_dimensions, n_train, n_test

def print_mnist_examples():
	""" Print some examples of the MNIST database in human-readable format """
	print("Let's print the first few images and labels of the training set")
	
	for case in range(5):
		for i in range(28):
			for j in range(28):
				pixel_index = 28 * i + j
				print( "{:3}".format(mnist.x_train[case, pixel_index]), end=" ") 
			print()

		print("Label: " + str(mnist.y_train[case,0]))
		print()

	print()
	print("Let's print the first few images and labels of the test set")
	for case in range(5):
		for i in range(28):
			for j in range(28):
				pixel_index = 28 * i + j
				print( "{:3d}".format(mnist.x_test[case, pixel_index]), end=" ") 
			print()

		print("Label: " + str(mnist.y_test[case,0]))
		print()

def main():
	## Load the training and test data into arrays 
	print_mnist_examples()

if __name__ == "__main__":
	main()
