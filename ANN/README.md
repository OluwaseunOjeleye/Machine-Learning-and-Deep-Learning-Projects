# Neural Network in C++

Implementation of neural network in C++ using matrices. This implementation is meant to be used for learning about neural networks and deep learning.

<p align="center">
  <img src="include/ANN.png" title="ANN" alt="ANN">
</p>

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
Linux Machine with gcc compiler.

### Downloading
Cloning The GitHub Repository

```
git clone 
```
### Neural Network Methods

NeuralNetwork(vector no_of_layers, string h_Layer_Activation, string output_Layer_Activation): Initializing NeuralNetwork

NeuralNetwork(string filename): Initializing Neural Network with already saved Neural Network's parameters

Train(vector X_train, vector Y_train, double alpha): Training a single data

Matrix Predict(vector X_test): Making Prediction

save_NeuralNetwork(string filename): Saving Neural Network to file

```
Example:
std::vector<int> layers={3, 2, 5}: three hidden layers excluding ouput layer with 3, 2 and 5 neurons.

NeuralNetwork Network(layers, relu, sigmoid): Network hidden layers activation function is relu and output layer function is sigmoid.

NeuralNetwork Network("filepath"): initialize Network with parameters in file.

Network.Train(X_train, Y_train, learning_rate): Training a single data in Network.

Network.Predict(X_test): Predict Y value for data X_test.

Network.save_NeuralNetwork("filepath"): save Network parameters in file. 
```

### Compiling the program
Run this in the folder from the command-line:

```
make
./main
```

## Authors
* **Jamiu Oluwaseun Ojeleye** 