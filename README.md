# neural-network-implementation-for-MNIST-classification

A Python implementation of neural network for MNIST classification. Including single-layer, multi-layer perceptron, and CNN.
The CNN implementation consists of a convolutional layer followed by a ReLu activation layer and a pooling layer. The output tensor was then flattened and passed through fully connected layer. The loss measures the cross-entropy of the soft-max output from the fully connected layer. 

![slp_linear](/result/slp_linear.PNG)
![slp](/result/slp_per.PNG)
![mlp](/result/mlp.PNG)
![cnn](/result/cnn.PNG)

## Dependencies
* Numpy
* matplotlib
* scipy 1.2.1

## Acknowledgment
Visualization code and data are from UMN Fall 2019 CSCI 5561 course material.
