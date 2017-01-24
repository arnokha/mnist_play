## Loosely adapted from http://pjreddie.com/projects/mnist-in-csv/
## Author: Arno Khachatourian

import numpy as np

n_train = 60000
n_test = 10000
img_dimensions = 28 * 28

x_train = np.zeros((n_train, img_dimensions), dtype=np.uint8)
y_train = np.zeros((n_train, 1), dtype=np.uint8)
x_test = np.zeros((n_test, img_dimensions), dtype=np.uint8)
y_test = np.zeros((n_test, 1), dtype=np.uint8)

def data2arrays(image_file, label_file, x, y, n):
	""" Extract the training and test data into numpy arrays """
	# Open files for reading
	img_reader = open(image_file, "rb")
	lbl_reader = open(label_file, "rb")

	img_reader.read(16) ## offset - first x bytes are the magic number, n, and dimensions of the image data
	lbl_reader.read(8)  ## offset - first x bytes are the magic number and n
	images = []

	for i in range(n):
		y[i] = ord(lbl_reader.read(1)) ## Read in one byte and transform to unicode
		for j in range(img_dimensions): ## Read next 256 bytes and transfer them to unicode
			x[i,j] = ord(img_reader.read(1))
	
	return (x,y)

## Load the training and test data into arrays 
print("Initializing... Loading training and test data...")
(x_train, y_train) = data2arrays("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte", x_train, y_train, n_train)
(x_test, y_test) = data2arrays("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte", x_test, y_test, n_test)
print("Finished loading data")

##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=##
### This part is strictly for visualization purposes
##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=##
print("This is just to test that the data loader worked correctly")
print("Let's print the first few images and labels of the training set")
for case in range(5):
	for i in range(28):
		for j in range(28):
			pixel_index = 28 * i + j
			print( "{:3}".format(x_train[case, pixel_index]), end=" ") 
		print()

	print("Label: " + str(y_train[case,0]))
	print()

print()
print("Let's print the first few images and labels of the test set")
for case in range(5):
	for i in range(28):
		for j in range(28):
			pixel_index = 28 * i + j
			print( "{:3d}".format(x_test[case, pixel_index]), end=" ") 
		print()

	print("Label: " + str(y_test[case,0]))
	print()