import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load the .npz file
with np.load("mnist.npz") as data:
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]

# Normalize and flatten
x_train = x_train.reshape(-1, 28 * 28).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype(np.float32) / 255.0

# One-hot encode labels
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

class model():
    def __init__(self):
        self.model = []

    def layer(self, number_of_nodes, activation):
        return [number_of_nodes, activation]
    
    def network(self, list_of_layers_and_activations):
        for layer in range(len(list_of_layers_and_activations)-1):
            self.model.append([np.random.uniform(-1, 1, (list_of_layers_and_activations[layer][0], list_of_layers_and_activations[layer+1][0])),    # weight's matrix
                               np.random.uniform(-0.5, 0.5, (1, list_of_layers_and_activations[layer+1][0]))                                   ,    # biase's vector
                               list_of_layers_and_activations[layer+1][1]])                                                                         # activation (firs one not concidered as it's input layer)
        return self.model
    
    def save(self, path, network):
        with open(path, "wb") as file:
            pickle.dump(network, file)

    def load(self, path):
        with open(path, "rb") as file:
            return pickle.load(file)


class Softmax():
    def function(self, input):
        exp = np.exp(input)
        return exp/np.sum(exp)
    
    def derivative(self, input):
        return 1 #self.function(input)

class ReLu():
    def function(self, input):
        return np.maximum(0, input)
    
    def derivative(self, input):
        return np.where(input > 0, 1.0, 0.0)

class Sigmoid():
    def function(self, input):
        return 1 / (1 + np.exp(-input))
    
    def derivative(self, input):
        return self.function(input)*(1 - self.function(input))
    
class MSE():
    def function(self, y, y_hat):
        return np.mean((y_hat-y)**2)
    
    def derivative(self, y, y_hat):
        return y_hat - y
    
class Cross_Entropy():
    def function(self, y, y_hat):
        return -np.sum(y * np.log(y_hat + 1e-9)) / y.shape[0]
    
    def derivative(self, y, y_hat):
        return y_hat - y

def save(path, network):
    with open(path, "wb") as file:
        pickle.dump(network, file)

def load(path):
    with open(path, "rb") as file:
        return pickle.load(file)

def eval(input, model):
    input_tracker = [input]
    for layer in model:
        input = layer[2].function(np.matmul(input, layer[0]) + layer[1])
        input_tracker.append(input)

    return input, input_tracker[:-1]

def backprop(y, y_hat, input_tracker, model, loss_function, LR):
    ### last layer ###

    current_layer_error = loss_function.derivative(y, y_hat)*model[-1][2].derivative(y_hat)     # output layer error gradient
    gradients = []
    for i in range(len(model)):
        layer = len(model) - i - 1      # going thru layers backwards

        #print(input_tracker[layer].T)
        output_layer_weights_gradient = np.matmul(input_tracker[layer].T, current_layer_error)
        output_layer_biases_gradient = current_layer_error
        gradients.append([output_layer_weights_gradient, output_layer_biases_gradient])

        # setting up new layer error for next layer
        current_layer_error = (np.matmul(model[layer][0], current_layer_error.T)*model[layer-1][2].derivative(input_tracker[layer].T)).T
    
    gradients = gradients[::-1]

    for gradient, layer in zip(gradients, model):
        layer[0] -= gradient[0]*LR
        layer[1] -= gradient[1]*LR

    return gradients, loss_function.function(y, y_hat)

def train(epochs, learning_rate, loss_function, my_model, samples, data):
    for epoch in range(epochs):
        print('epoch', epoch+1)
        counter = 0
        total_loss = []
        for sample, target in zip(samples, data):

            y_hat, input_tracker = eval(np.array([sample]), my_model)
            gradients, loss = backprop(np.array([target]), y_hat, input_tracker, my_model, loss_function, learning_rate)

            total_loss.append(loss)
            training = 100*counter/len(samples)
            counter += 1

            if round(training, 2)%10 == 0:
                print(f"Progress: {str(training)[:4]}%     Loss: {sum(total_loss)/len(total_loss)}" , end='\r')
        print()
 
model = model()

my_model = model.network([
    model.layer(784, None),
    model.layer(128, ReLu()),
    model.layer(128, ReLu()),
    model.layer(10, Softmax())
])

my_model = model.load('mnist_model.pkl')

#train(12, 0.005, Cross_Entropy(), my_model, x_train, y_train)

# accuracy check
accuracy = 0
for sample, target in zip(x_train, y_train):
    y_hat, tracker = eval(np.array([sample]), my_model)
    if np.argmax([target]) == np.argmax([y_hat]):
        accuracy += 1

save("mnist_model.pkl", my_model)
print('accuracy:', accuracy/len(x_train), '%')
