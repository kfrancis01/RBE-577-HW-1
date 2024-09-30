# RBE-577-HW-1 
# Control Allocation via Deep Neural Networks

## Prerequisites

* **Visual Studio Code** (Version 1.93)
* **Python** (Version 3.9.13)
* **PyTorch** (Version 2.2.2)
* **NumPy** (Version 1.23.5)
* **Matplotlib** (Version 3.5.2)
* **Reference Research Paper** (Constrained control allocation for dynamic ship positioning using deep neural network)

## Getting Started

1. Clone this repository.
2. Open the project folder in Visual Studio Code.
3. Make sure you have the required libraries installed. 
```pip install torch numpy matplotlib```
4. Run the main Python script

## Methodology
* **Data Generation:**
    * Synthetic data was generated to represent the control allocation problem based on the force and angle ranges specified in the research paper (page 3).
      ```
      F1_range = [-10000, 10000]
      F2_range = [-5000, 5000]
      alpha2_range = [-180, 180]
      F3_range = [-5000, 5000]
      alpha3_range = [-180, 180]
      ```
    * The dataset was split into training and testing sets using an 80/20 ratio.
    * The generated data is used as inputs for the deep neural network (DNN) to learn control allocation for ship positioning.

* **Deep Neural Network (DNN) Model:**
    * 3 encoder hidden layers for processing inputs.
    * 3 decoder hidden layers for reconstructing inputs.
    * Each hidden layer has batch normalization, dropout, and ReLU activation functions for better convergence and overfitting prevention.
    * The model and hyperparameters were adjusted iteratively to improve performance.

* **Training and Testing:**
    * **Training** \
      ** The model was trained on the training data using backpropagation and an SGD optimizer.
      ** Iterative updates to improve the curves by reducing overfitting noise and aid attempts to converge to zero effectively.
      ** Regularization techniques such as dropout and weight decay are used to reduce overfitting.
    * **Testing** \
      ** The testing loss is calculated and plotted to visualize how well the model performs on unseen data.
      
## Hyperparameter Adjustments
* **Learning Rate Decay**
    * The learning rate decay was lowered from an original value of 0.1 to 0.001 in order to prevent NaN loss values
      ** Lowering the learning rate also reduced overfitting, allowing the model to converge more slowly and accurately.
      ** Per previous lectures it is best practice to update the learning rate after initial training

* **DNN Creation:**
    * The number of neurons in the hidden layers were iteratively adjusted to improve the speed and performance of the model.
    * Batch Size: Adjustments were made to batch size for training and testing
      ** A larger batch size was used during testing to ensure stability and accuracy.
    * I adjusted the hidden layer configuration in order to escalate the graph convergence. 
      ** adjustments would be an increase by x2 for consistency. 
      
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
   ** Normalization addresses overfitting errors.
   ** Input data was normalized to ensure stable training and better convergence.
  ```
   F1 = np.random.uniform(F1_range[0], F1_range[1], num_samples) / 10000  # Normalize to [-1, 1]
   F2 = np.random.uniform(F2_range[0], F2_range[1], num_samples) / 5000
   F3 = np.random.uniform(F3_range[0], F3_range[1], num_samples) / 5000
   alpha2 = np.random.uniform(alpha2_range[0], alpha2_range[1], num_samples) / 180
   alpha3 = np.random.uniform(alpha3_range[0], alpha3_range[1], num_samples) / 180
  ```

* **Dropout** 

  ** Applied during training and testing to reduce overfitting.
  ** Ensures all nodes contribute to learning and prevents over-reliance on specific neurons.
  ** Initial dropout rate of 0.2 was increased to 0.5 when overfitting was observed, resulting in smoother curves.

* **Number of epochs**
  ** Incrementally increased to allow the model sufficient training until average loss values approached zero.
  ** Ensured the model had enough training to converge effectively.
  
* **Batch Normalization**
  ** With batch normalization weight intialization wasn't considered 
  ** Stabilized and accelerated training
  ** The number of training epochs was adjusted to ensure the model converged to a low loss value.

* **Weight Decay** \
  ** Weight decay was increased from 1e-2 in order to help address overfitting mostly happening in the testing process. Increasing the weight decay helped get training and testing loss values to match more.
  ** Weight decay was increased to penalize large weights and further combat overfitting, helping to align the training and testing loss curves.

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

## Sample Output Graph \
Below is an example of the loss vs. epoch graph, showing the training and testing loss values over the course of model training.

![alt text](image.png)

