# RBE-577-HW-1 Control Allocation via Deep Neural Networks

## Prerequisites
##### Visual Studio Code Version 1.93
##### Python Version 3.9.13
##### Torch Version 2.2.2
##### Numpy Version 1.23.5
##### Matplotlib Version 3.5.2

## Getting Started
1. Open Visual Studio Code
2. Import the python file
3. Run the program

## Methodology

## Hyperparameter Adjustments
#### Learning Rate Decay
The learning rate decay was lowered from an original value of 0.1 to 0.001 in order to prevent NaN loss values 

#### input and output parameters for DNN creation
I adjusted the metrics in order to escalate the graph getting to zero. I would increase everything x2 during the adjustments.
```
num_hidden1 = 512
num_hidden2 = 1024
num_hidden3 = 2048
```

## Data Generation
### Using an 80/20 percent split
As instructed the data was split with 80% being used for training and 20% used for testing
Split the dataset into training and testing sets (80/20)
```
split = int(0.8 * num_samples)
X_train, X_test = X_train[:split], X_train[split:]
Y_train, Y_test = Y_train[:split], Y_train[split:]
```

## Regularization Techniques
#### Normalizing Data
The data was normalized inorder to

#### Dropout
Dropout was used in the training and testing DNN model generation in order to reduce overfitting by allowing each node to be utlized equally. The end result was a smoother curve. 

#### Number of epochs
The number of epochs were increased until there were enough for the average loss values to reach close to zero

#### Batch Normalization
With batch normalization weight nitialization wasn't considered 

## Lessons Learnt 

