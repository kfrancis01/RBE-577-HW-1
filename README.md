# RBE-577-HW-1 Control Allocation via Deep Neural Networks

## Prerequisites

* **Visual Studio Code** (Version 1.93)
* **Python** (Version 3.9.13)
* **PyTorch** (Version 2.2.2)
* **NumPy** (Version 1.23.5)
* **Matplotlib** (Version 3.5.2)

## Getting Started

1. Clone this repository.
2. Open the project folder in Visual Studio Code.
3. Make sure you have the required libraries installed (```pip install torch numpy matplotlib```).
4. Run the main Python script

## Methodology
* **Data Generation:**
    * Synthetic data was generated to represent the control allocation problem.
    * The dataset was split into training and testing sets using an 80/20 ratio.

* **Deep Neural Network (DNN) Model:**
    * A DNN model was designed with 3 encoder hidden layers, 3 decoder hidden layers, and activation functions.
    * The model and hyperparameters were adjusted iteratively to improve performance.

* **Training and Testing:**
    * The model was trained on the training data using backpropagation and an optimizer.
    * Iterative updates to improve the curves reducing noise and converging to zero effectively
      
## Hyperparameter Adjustments
* **Learning Rate Decay**
    * The learning rate decay was lowered from an original value of 0.1 to 0.001 in order to prevent NaN loss values

* **DNN Creation:**
    * The number of neurons in the hidden layers was iteratively adjusted to improve the analysis speed.
    * I adjusted the metrics in order to escalate the graph getting to zero. I would increase everything x2 during the adjustments.
```
num_hidden1 = 512
num_hidden2 = 1024
num_hidden3 = 2048
```

## Data Generation
* **Using an 80/20 percent split**
As instructed the data was split with 80% being used for training and 20% used for testing
Split the dataset into training and testing sets (80/20)

```
split = int(0.8 * num_samples)
X_train, X_test = X_train[:split], X_train[split:]
Y_train, Y_test = Y_train[:split], Y_train[split:]
```

## Regularization Techniques
* **Normalizing Data**
The data was normalized inorder to
* The input data was normalized to improve the stability and convergence of the training process.

* **Dropout**
Dropout was used in the training and testing DNN model generation in order to reduce overfitting by allowing each node to be utlized equally. The end result was a smoother curve. The dropout parameter was initially at 0.2; however, when overfitting was noticed on the testing loss plot the parameter was increased in order to address it.
    * Dropout layers were added to prevent overfitting by randomly deactivating neurons during training. 
    * The dropout rate was increased to further mitigate overfitting observed in the testing loss.

* **Number of epochs**
The number of epochs were increased until there were enough for the average loss values to reach close to zero
* The number of training epochs was adjusted to ensure the model converged to a low loss value.
  
* **Batch Normalization**
With batch normalization weight nitialization wasn't considered 
* The number of training epochs was adjusted to ensure the model converged to a low loss value.

* **Weight Decay**
Weight decay was increased from 1e-2 in order to help address overfitting mostly happening in the testing process. Increasing the weight decay helped get training and testing loss values to match more.
* Weight decay was increased to penalize large weights and further combat overfitting, helping to align the training and testing loss curves.

## Lessons Learnt 
* Batch size plays a role in both training and testing performance.
  ** Batch size needed to be higher for testing but lower for training
* Dropout is effective in reducing overfitting, especially in the testing phase.
  ** for dropout to be effective in needed to be in the 0.4 - 0.6 range originally I had it at 0.2 
* Overfitting can be more pronounced in the testing set than in the training set.
* Learning rate decay is crucial for stable training and can impact both training and testing loss.
* Increasing the learning rate can help overcome overfitting but might slow down convergence.\
* * Difficult to make updates since something that benefits testing might hurt the training. For example increasing the batch size benefits testing but increases the time for convergence when training.

