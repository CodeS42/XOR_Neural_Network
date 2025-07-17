import numpy as np
import calculations
import XOR

class NeuralNetwork:
    def __init__(self):
        self.hidden_weights = np.random.rand(2, 2)
        self.hidden_bias = np.random.rand(2)
        self.output_hidden_weights = np.random.rand(2)
        self.output_bias = np.random.rand()
    
    def forward_propagation(self, inputs):
        z = calculations.linear_activation(inputs, self.hidden_weights, self.hidden_bias)
        a = calculations.sigmoid(z)

        z = calculations.linear_activation(a, self.output_hidden_weights, self.output_bias)
        a = calculations.sigmoid(z)

        return a

    def train(self):
        for inputs in XOR.inputs:
            output = self.forward_propagation(inputs)