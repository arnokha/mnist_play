##
## Average the cases in the training set to create a "mold" to use for predictions
## Print those molds
##
## Author: Arno Khachatourian
##

import numpy as np
import create_digit_molds

def print_molds(molds):
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
	molds = create_digit_molds.create_molds()
	print_molds(molds)

## Main
if __name__ == "__main__":
	main()

