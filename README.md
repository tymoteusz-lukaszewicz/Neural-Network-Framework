# Neural-Network-Framework

# Simple Fully Connected Neural Network Framework for MNIST Classification

## Purpose

Presentation of the final version of the project for my portfolio. This project implements a basic framework for training and evaluating fully connected neural networks. It demonstrates the fundamental concepts of neural network architecture, activation functions, loss functions, forward propagation, and backpropagation. The network is trained on the MNIST handwritten digit dataset.

## Files in Repository

* `fully_connected_network.py`: The Python script containing the implementation of the neural network framework and the training/evaluation process.
* `mnist.npz`: The MNIST dataset in NumPy's `.npz` format, containing training and testing images and labels.
* `mnist_model.pkl`: A pre-trained model (may or may not be present initially, will be created after training).

## Technologies Used

* Programming Language: Python
* Libraries: NumPy, Matplotlib, Pickle

## Setup Instructions

1.  Make sure you have Python 3.x installed.
2.  Install the required libraries if you haven't already:
    ```bash
    pip install numpy matplotlib
    ```
3.  Download the `mnist.npz` file (it should be included in the repository).
4.  The `mnist_model.pkl` file will be created after you train the network.

## Running the Code

To train and evaluate the neural network, simply run the Python script:

```bash
python fully_connected_network.py
