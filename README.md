# Home_Assignment_1

# Question 1
Create a Random Tensor:
Makes a random grid of numbers with shape (4, 6).

1.Check Rank and Shape:
  Rank tells the number of dimensions (like rows and columns).
  Shape shows the size of each dimension (here, 4 rows and 6 columns).

2.Reshape and Transpose:
  Reshape changes it to (2, 3, 4) (2 blocks, 3 rows, 4 columns).
  Transpose swaps dimensions to (3, 2, 4).

3.Broadcast and Add:
  Takes a smaller tensor of shape (1, 4).
  Broadcasts it to match the larger tensor’s shape.
  Adds the two tensors together.

# Question 2
Comparing Loss Functions: MSE vs. Categorical Cross-Entropy:
This code compares two loss functions:
-Mean Squared Error (MSE): Measures the average squared difference between true and predicted values.
-Categorical Cross-Entropy (CCE): Measures error for predicted probabilities in multiple classes.

Steps:
1.Define True Values and Predictions
  y_true = [0, 1, 0]
  y_pred = [0.2, 0.7, 0.1]

2.Calculate Loss Values
  MSE and CCE are calculated using custom functions.
  Loss values are printed.
  
3.Modify Predictions
  Slightly change predictions to [0.3, 0.6, 0.1].
  Recalculate and print new loss values.
  
4.Visualize Loss Values
  Bar chart comparing initial and modified MSE and CCE values using Matplotlib.

* Observations:
  MSE increases more with larger errors.
  CCE increases when wrong predictions are made with high confidence.
  Small changes in predictions affect MSE and CCE differently.

# Question 3
Training MNIST Model with Different Optimizers: SGD vs. Adam:
* This code compares the performance of two optimizers—SGD and Adam—on the MNIST dataset.

Steps Involved:
1.Load the MNIST Dataset:
  Load training and test data for handwritten digit recognition.

2.Preprocess the Data:
  Normalize pixel values to be between 0 and 1.
  Expand the shape of input images to include a channel dimension.
  Convert labels to categorical (one-hot encoding).
  
3.Build the Neural Network Model:
  A simple CNN model with 1 convolutional layer, 1 pooling layer, and 2 fully connected layers.
  
4.Train the Model with SGD:
  Compile the model using the SGD optimizer.
  Train for 10 epochs and track training/validation accuracy.
  
5.Train the Model with Adam:
  Compile the model using the Adam optimizer.
  Train for 10 epochs and track training/validation accuracy.
  
6.Plot Accuracy Trends:
  Compare training and validation accuracy of both optimizers using a line plot.
  
* Results:
  The plot shows the training and validation accuracy for both SGD and Adam optimizers.
  Adam is typically faster and shows better performance compared to SGD.

# Question 4
Train a Neural Network with TensorBoard Logging:
*This code trains a simple neural network on the MNIST dataset and logs the training process to TensorBoard for analysis.

Steps Involved:
1.Import Libraries:
  Import necessary libraries like TensorFlow, Keras, and MNIST dataset.
  Load and Preprocess MNIST Data:

2.Load the MNIST dataset (handwritten digit images).
  Flatten images to a 1D array of 784 pixels and normalize values to be between 0 and 1.
  
3.Build Neural Network Model:
A simple model with one hidden layer (128 neurons, ReLU activation) and an output layer with 10 neurons (softmax activation for multi-class classification).

4.Compile the Model:
  Use Adam optimizer, sparse_categorical_crossentropy loss function, and track accuracy.
  
5.Set Up TensorBoard Logging:
  Log the training process using TensorBoard to visualize performance. Logs are stored in logs/fit/ folder.

6.Train the Model for 5 Epochs:
  Train the model on the MNIST dataset for 5 epochs and validate on test data.

7. Visualistion:
   By using following commands %load_ext tensorboard and %tensorboard --logdir logs/fit we can be able to visualize training vs. validation accuracy and loss in 
TensorBoard.
