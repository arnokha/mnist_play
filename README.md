# mnist_play

## Overivew of this project

This project is meant for me to practice as many interesting ML techniques as I can on the MNIST data set.

I intend to clean and organize this repository as I progress with this project.

## Performance of methods

For each method used, I will record any metrics that I deem relevant or interesting. 

### Method: Mold matching

#### Mold matching (results from test_molds.py)

Total training time (time to create molds) was: 1.0927739579929039 seconds  
Total test time was (on training data): 6.7956149550154805 seconds  
Total test time was (on test data): 1.1567412989679724 seconds  

Error rate on the training data was: 0.3768  
**Error rate on the test data was: _0.369_**

#### Mold matching (with regularized pixel intensities) (results from test_regularized_molds.py)

Total training time (time to create regularized molds) was: 1.0254628909751773 seconds  
Total test time was (on training data): 6.2145653740735725 seconds  
Total test time was (on test data): 1.004182494012639 seconds  

Error rate on the training data was: 0.37868333333333337  
**Error rate on the test data was: _0.3669_**

#### k-NN (k = 245) (distance = sum(abs(x1 - x2)))

Total training time (time to create molds) was: 5543.033263341989 seconds  
Total test time was (on test data): 0.0413792310282588 seconds  

**Error rate on the test data was: 0.240**

#### k-NN best results (Using evaluate_k_NN_from_files.py)

The value of k that produced the lowest error rate was 10  
The accuracy on the test set was 0.8249

**The error on the test set was 0.1751**
