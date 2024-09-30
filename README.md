# RBE-577-HW-1 
# Control Allocation via Deep Neural Networks

## Prerequisites

* **Visual Studio Code** (Version 1.93)
* **Python** (Version 3.9.13)
* **PyTorch** (Version 2.2.2)
* **NumPy** (Version 1.23.5)
* **Matplotlib** (Version 3.5.2)
* **Constrained control allocation for dynamic ship positioning using deep neural network** (Reference Research Paper)

## Getting Started

1. Clone this repository.
2. Open the project folder in Visual Studio Code.
3. Make sure you have the required libraries installed (```pip install torch numpy matplotlib```).
4. Run the main Python script

## Methodology
* **Data Generation:**
    * Synthetic data was generated to represent the control allocation problem.
    * Ranges used were found on pg. 3 of the research paper
      ```
      F1_range = [-10000, 10000]
      F2_range = [-5000, 5000]
      alpha2_range = [-180, 180]
      F3_range = [-5000, 5000]
      alpha3_range = [-180, 180]
      ```
    * The dataset was split into training and testing sets using an 80/20 ratio.

* **Deep Neural Network (DNN) Model:**
    * A DNN model was designed with 3 encoder hidden layers, 3 decoder hidden layers, and activation functions that allowed the model to understand the sequence.
    * The model and hyperparameters were adjusted iteratively to improve performance.

* **Training and Testing:**
    * The model was trained on the training data using backpropagation and an SGD optimizer.
    * Iterative updates to improve the curves by reducing overfitting noise and aid attempts to converge to zero effectively.
      
## Hyperparameter Adjustments
* **Learning Rate Decay**
    * The learning rate decay was lowered from an original value of 0.1 to 0.001 in order to prevent NaN loss values
      ** This change also helped with reducing overfitting since a lower learning rate reduces the speed of convergence allowing for a more precise value
      ** Per previous lectures it is best practice to update the learning rate after initial training

* **DNN Creation:**
    * The number of neurons in the hidden layers were iteratively adjusted to improve the speed and performance of the training and test loss convergence.
    * I adjusted the metrics in order to escalate the graph getting to zero. I would increase everything x2 during the adjustments. 
      
      ```
      num_hidden1 = 512
      num_hidden2 = 1024
      num_hidden3 = 2048
      ```

## Data Generation
* **Using an 80/20 percent split** \
As instructed the data was split with 80% being used for training and 20% used for testing
   
   ```
   split = int(0.8 * num_samples)
   X_train, X_test = X_train[:split], X_train[split:]
   Y_train, Y_test = Y_train[:split], Y_train[split:]
   ```

## Regularization Techniques

* **Normalizing Data** \
   ** The data was normalized inorder to address overfitting.
   ** The input data was normalized to improve the stability and convergence of the training process.
  ```
   F1 = np.random.uniform(F1_range[0], F1_range[1], num_samples) / 10000  # Normalize to [-1, 1]
   F2 = np.random.uniform(F2_range[0], F2_range[1], num_samples) / 5000
   F3 = np.random.uniform(F3_range[0], F3_range[1], num_samples) / 5000
   alpha2 = np.random.uniform(alpha2_range[0], alpha2_range[1], num_samples) / 180
   alpha3 = np.random.uniform(alpha3_range[0], alpha3_range[1], num_samples) / 180
  ```

* **Dropout** \
Dropout was used in the training and testing DNN model generation in order to reduce overfitting by allowing each node to be utlized equally. The end result was a smoother curve. The dropout parameter was initially at 0.2; however, when overfitting was noticed on the testing loss plot the parameter was increased in order to address it.
    * Dropout layers were added to prevent overfitting by randomly deactivating neurons during training. 
    * The dropout rate was increased to further mitigate overfitting observed in the testing loss.

* **Number of epochs** \
The number of epochs were increased until there were enough for the average loss values to reach close to zero
* The number of training epochs was adjusted to ensure the model converged to a low loss value.
  
* **Batch Normalization** \
With batch normalization weight nitialization wasn't considered 
* The number of training epochs was adjusted to ensure the model converged to a low loss value.

* **Weight Decay** \
Weight decay was increased from 1e-2 in order to help address overfitting mostly happening in the testing process. Increasing the weight decay helped get training and testing loss values to match more.
* Weight decay was increased to penalize large weights and further combat overfitting, helping to align the training and testing loss curves.

## Lessons Learnt 
* Batch size plays a role in both training and testing performance.
  ** Higher for testing, lower for training: A larger batch size during testing can lead to more stable and accurate evaluations, while a smaller batch size during training can introduce more noise
* Dropout is effective in reducing overfitting, especially in the testing phase.
  ** Effective range: 0.4 - 0.6: Experimentation showed that dropout rates within this range were most effective in mitigating overfitting especially for testing.
* Overfitting can be more pronounced in the testing.
  ** The model might perform well on the training data but struggle when it comes to testing data
* Learning rate is important for stable training.
  ** Effective range: 1e-3 to 1e-4: Experimentation showed that learning rate decay values within this range were most effective in getting earlier convergence.
  ** Increasing the learning rate can help overcome overfitting but might slow down convergence.\
* Weight decay had an effective range of 1e-3 to 1e-4 and would help to make the training and test loss closer together
* Difficult to make updates since something that benefits testing might hurt the training. For example increasing the batch size benefits testing but increases the time for convergence when training.
  ** It is best to do one parameter at a time wait until the graphs get as close to the desired result as possible than move to another parameter

