## Use kNN for mnist

import numpy as np
import extract_mnist_data as mnist
## Now have access to x_train, y_train, x_test, y_test, n_train, n_test, img_dimensions, and n_labels

#def weight():

def distance(x1, x2):
	return np.sum(abs(x1 - x2))

def kNN(k):
	pass

def NN():
	k = 1
	knn_x_train = np.zeros((mnist.n_train, k), dtype=np.uint8)
	d = np.zeros((mnist.n_train, mnist.n_train), dtype=np.uint16)
	max_distance = 2 ** 16 - 1

#	for i in range(mnist.n_train):
	for i in range(50):
		for neighbor in range(mnist.n_train):
			d[i,neighbor] = distance( mnist.x_train[i,:], mnist.x_train[neighbor,:] )

		d[i,i] = max_distance ## Commenting this out should make this 100% accturate on training set when k = 1
#		if i % 1000 == 0:
#			print(str(i))
		for k_index in range(k):
			index_min_distance = np.argmin(d[i,:])
			d[i,index_min_distance] = max_distance ## Set to max so next iteration wont pick again... need to remove if I'm gonna use weighting
			knn_x_train[i,k_index] = index_min_distance

		print( "Nearest neighbor of x_train #" + str(i) + " is " + str(knn_x_train[i]) )
		print( "Label of nearest neighbor of x_train is " + str( mnist.y_train[ knn_x_train[i] ] ) )
		print( "Actual label of x_train #" + str(i) + " is " + str( mnist.y_train[i] ) )

#	for i in range(mnist.n_train):
#		if i % 1000 == 0:
#			print( "Nearest neighbor of x_train #" + str(i) + " is " + str(knn_x_train[i,k]) )
#			print( "Label of nearest neighbor of x_train is " + str( mnist.y_train[ knn_x_train[i,k] ] ) )
#			print( "Actual label of x_train #" + str(i) + " is " + str( mnist.y_train[i] ) )

def main():
	NN() ## This is just to help me implement kNN

	exit() ## TODO remove garbage
	k = int(input("Enter an odd number to use for k-Nearest-Neighbors: "))
	if ( k % 2 == 0): 
		## Number is even. Prompt user for an odd number
		print(str(k) + " is an even number. Please enter an odd number.")
		exit()
	kNN(k)

	print("Testing distance function...") ##TODO remove following lines
	x1 = np.ones((2,2))
	x2 =  - np.ones((2,2))
	print(str(distance(x1,x2)))

if __name__ == "__main__":
	main()
