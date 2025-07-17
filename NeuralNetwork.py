import numpy as np
import calculations
import XOR

class NeuralNetwork:
    def __init__(self):
        self.hidden_weights = np.random.rand(2, 2)
        self.hidden_bias = np.random.rand(2)
        self.output_hidden_weights = np.random.rand(2)
        self.output_bias = np.random.rand()
    
    def forward_propagation(self, input1, input2):
        z = calculations.linear_activation(self.hidden_weights, self.hidden_bias, input1, input2)
        a1 = calculations.sigmoid(z[0])
        a2 = calculations.sigmoid(z[1])

        z = calculations.linear_activation_output(self.output_hidden_weights, self.output_bias, a1, a2)
        a = calculations.sigmoid(z)

        return a

    def train(self):
        for inpt in XOR.inputs:
            output = self.forward_propagation(inpt[0], inpt[1])