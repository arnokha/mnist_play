##
## Average the cases in the training set to create a "mold" to use for predictions
## Print those molds
##
## Author: Arno Khachatourian
##

import numpy as np
from extract_mnist_data import data2arrays
from create_digit_molds import create_molds, count_labels

def print_molds():
	""" Print the molds created by create_digit_molds.py """
	n_labels = 10

	## Print digit molds to files "{n}.mold" just to see what they look like (not for classification use)
	print("Let's print the molds for each label...")
	for label in range(n_labels):
		for i in range(28):
			for j in range(28):
				pixel_index = (28 * i) + j
				print( "{:3d}".format(molds[label, pixel_index]), end=" ") 
			print()

		print( "Label: " + str(label) )
		print()

def main():
	molds = create_molds()
	print_molds()

## Main
if __name__ == "__main__":
	main()

