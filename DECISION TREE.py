import math
import random

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class FeedForwardNN:
    def __init__(self, num_inputs, num_hidden_units, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden_units = num_hidden_units
        self.num_outputs = num_outputs

        # Initialize weights with random values between 0 and 1
        self.weights_input_hidden = [[random.random() for _ in range(self.num_hidden_units)] for _ in range(self.num_inputs)]
        self.weights_hidden_output = [[random.random() for _ in range(self.num_outputs)] for _ in range(self.num_hidden_units)]

    def dot_product(self, inputs, weights):
        # Perform dot product of input vector and weight matrix
        result = []
        for j in range(len(weights[0])):
            sum_product = 0
            for i in range(len(inputs)):
                sum_product += inputs[i] * weights[i][j]
            result.append(sum_product)
        return result

    def forward_pass(self, inputs):
        # Hidden layer activations
        hidden_input = self.dot_product(inputs, self.weights_input_hidden)
        self.hidden_activation = [sigmoid(x) for x in hidden_input]

        # Output layer activations
        output_input = self.dot_product(self.hidden_activation, self.weights_hidden_output)
        self.output_activation = [sigmoid(x) for x in output_input]

        return self.output_activation

# Parameters
num_inputs = 3
num_hidden_units = 4
num_outputs = 2

# Create model instance
model = FeedForwardNN(num_inputs, num_hidden_units, num_outputs)

# Input data (random values for testing)
input_data = [random.random() for _ in range(num_inputs)]

# Forward pass
output_data = model.forward_pass(input_data)

print("Output:", output_data)
