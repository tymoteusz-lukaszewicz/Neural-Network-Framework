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
    # initializing model object
    def __init__(self):
        self.model = []
        
    # defining layer structure
    def layer(self, number_of_nodes, activation):
        return [number_of_nodes, activation]

    # creating network out of many layers
    def network(self, list_of_layers_and_activations):
        for layer in range(len(list_of_layers_and_activations)-1):
            self.model.append([np.random.uniform(-1, 1, (list_of_layers_and_activations[layer][0], list_of_layers_and_activations[layer+1][0])),    # weight's matrix
                               np.random.uniform(-0.5, 0.5, (1, list_of_layers_and_activations[layer+1][0]))                                   ,    # biase's vector
                               list_of_layers_and_activations[layer+1][1]])                                                                         # activation (first one not concidered as it's an input layer)
        return self.model

    # saving model
    def save(self, path, network):
        with open(path, "wb") as file:
            pickle.dump(network, file)

    # loading model
    def load(self, path):
        with open(path, "rb") as file:
            return pickle.load(file)

# softmax activation function
class Softmax():
    def function(self, input):
        exp = np.exp(input)
        return exp/np.sum(exp)
    
    def derivative(self, input):
        return 1
        
# relu activation function
class ReLu():
    def function(self, input):
        return np.maximum(0, input)
    
    def derivative(self, input):
        return np.where(input > 0, 1.0, 0.0)
        
# sigmoid activation function
class Sigmoid():
    def function(self, input):
        return 1 / (1 + np.exp(-input))
    
    def derivative(self, input):
        return self.function(input)*(1 - self.function(input))

# mean squared error loss function
class MSE():
    def function(self, y, y_hat):
        return np.mean((y_hat-y)**2)
    
    def derivative(self, y, y_hat):
        return y_hat - y

# cross entropy loss function
class Cross_Entropy():
    def function(self, y, y_hat):
        return -np.sum(y * np.log(y_hat + 1e-9)) / y.shape[0]
    
    def derivative(self, y, y_hat):
        return y_hat - y
# evaluating model (imput propagation)
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

        # calculating error gradient for current layer
        output_layer_weights_gradient = np.matmul(input_tracker[layer].T, current_layer_error)
        output_layer_biases_gradient = current_layer_error
        gradients.append([output_layer_weights_gradient, output_layer_biases_gradient])

        # setting up new layer error for next layer
        current_layer_error = (np.matmul(model[layer][0], current_layer_error.T)*model[layer-1][2].derivative(input_tracker[layer].T)).T
    
    gradients = gradients[::-1]

    # updating all gradients at once
    for gradient, layer in zip(gradients, model):
        layer[0] -= gradient[0]*LR
        layer[1] -= gradient[1]*LR

    return gradients, loss_function.function(y, y_hat)

# training loop
def train(epochs, learning_rate, loss_function, my_model, samples, data):
    for epoch in range(epochs):
        print('epoch', epoch+1)
        counter = 0
        total_loss = []

        # going thru all samples
        for sample, target in zip(samples, data):

            # evaluating model
            y_hat, input_tracker = eval(np.array([sample]), my_model)
            # updating weights
            gradients, loss = backprop(np.array([target]), y_hat, input_tracker, my_model, loss_function, learning_rate)

            # updating general networ loss
            total_loss.append(loss)
            training = 100*counter/len(samples)
            counter += 1
            
            if round(training, 2)%10 == 0:
                print(f"Progress: {str(training)[:4]}%     Loss: {sum(total_loss)/len(total_loss)}" , end='\r')
        print()

# Randomly selects a few test images, predicts their labels, and displays the images
# along with the true and predicted labels.
def visualize_predictions(model, x_test, y_test, num_samples=4):
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    fig, axes = plt.subplots(1, num_samples, figsize=(10, 3))

    for i, index in enumerate(indices):
        image = x_test[index].reshape(28, 28)
        true_label = np.argmax(y_test[index])
        predicted_label, _ = eval(np.array([x_test[index]]), model)
        predicted_label = np.argmax(predicted_label)

        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f"True: {true_label}, Predicted: {predicted_label}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# initializing network
model = model()

# setting up network's shape and activations
my_model = model.network([
    model.layer(784, None),
    model.layer(128, ReLu()),
    model.layer(128, ReLu()),
    model.layer(10, Softmax())
])

# optional: loading pre-trained model insted of creating from sctrath. After loading you can finish training or use model normally if fully trained.
# my_model = model()
# my_model = model.load('mnist_model.pkl')

# training
train(12, 0.005, Cross_Entropy(), my_model, x_train, y_train)

# accuracy check
accuracy = 0
for sample, target in zip(x_test, y_test):
    y_hat, tracker = eval(np.array([sample]), my_model)
    if np.argmax([target]) == np.argmax([y_hat]):
        accuracy += 1

# saving model
model.save("mnist_model.pkl", my_model)
print('validation set accuracy:', accuracy/len(x_test), '%')

# visualizing predictions
visualize_predictions(my_model, x_test, y_test)
