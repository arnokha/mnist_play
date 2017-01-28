## Average the cases in the training set to create a "mold" to use for predictions

import numpy as np
from extract_mnist_data import data2arrays

def count_labels( labels , n_labels ):
	""" Loop through 1D label array and count the instances of each label. Return count of each label in array form """
	label_count = np.zeros((n_labels, 1), dtype=np.uint16)

	for i in range(labels.size):
		label_count[ labels[i] ] += 1

	return label_count

def main():
	## Initialize data structures
	n_train = 60000
	n_test = 10000
	img_dimensions = 28 * 28
	
	x_train = np.zeros((n_train, img_dimensions), dtype=np.uint8)
	y_train = np.zeros((n_train, 1), dtype=np.uint8)
	x_test = np.zeros((n_test, img_dimensions), dtype=np.uint8)
	y_test = np.zeros((n_test, 1), dtype=np.uint8)

	n_labels = 10 # 0 - 9
	label_counter = np.zeros((n_labels, 1), dtype=np.uint16)  # Used for mold averaging
	molds = np.zeros((n_labels,img_dimensions), dtype=np.uint32)

	## Load data into data structures
	print("Loading data...")
	(x_train, y_train) = data2arrays("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", x_train, y_train, n_train)
	(x_test, y_test) = data2arrays("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte", x_test, y_test, n_test)
	
	label_counter = count_labels( y_train, n_labels )
	print("Here are the label counts... For fun and verification")
	print(label_counter)

	## Create molds (may need to use thresholds if numbers get too big)
	for i in range(n_train):
		molds[ y_train[i], : ] = np.add( molds[ y_train[i], : ], x_train[i, :] )

	## Regularize mold by averaging it by number of cases per label
	for i in range(n_labels):
		molds[i,:] = molds[i,:] / label_counter[i]

	## Print digit molds to files "{n}.mold" just to see what they look like (not for classification use)
	print()
	print("Let's print the first few images and labels of the test set")
	for label in range(n_labels):
		for i in range(28):
			for j in range(28):
				pixel_index = (28 * i) + j
				print( "{:3d}".format(molds[label, pixel_index]), end=" ") 
			print()

		print( "Label: " + str(label) )
		print()

## Main
if __name__ == "__main__":
	main()

