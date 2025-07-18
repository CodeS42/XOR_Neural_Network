import numpy as np
import calculations
import XOR

class NeuralNetwork:
    def __init__(self):
        self.hidden_weights = np.random.rand(2, 2)
        self.hidden_bias = np.random.rand(2)
        self.output_hidden_weights = np.random.rand(2)
        self.output_bias = np.random.rand()
        self.z_hidden = np.zeros((2))
        self.z_output = None
        self.a_hidden = np.zeros((2))
        self.a_output = None
        self.learning_rate = 0.01   
    
    def forward_propagation(self, inputs):
        self.z_hidden = calculations.linear_activation(inputs, self.hidden_weights, self.hidden_bias)
        self.a_hidden = calculations.sigmoid(self.z_hidden)

        self.z_output = calculations.linear_activation(self.a_hidden, self.output_hidden_weights, self.output_bias)
        self.a_output = calculations.sigmoid(self.z_output)
    
    def back_propagation(self, inputs, output):
        output_error = calculations.loss_function(self.a_output, output)
        output_delta = calculations.compute_output_delta(output_error, self.z_output)
        hidden_delta = calculations.backpropagate_error_to_hidden(output_delta, self.output_hidden_weights, self.z_hidden)
        self.output_hidden_weights -= calculations.update_weights(self.learning_rate, output_delta, self.a_hidden)
        self.hidden_weights -= calculations.update_weights(self.learning_rate, hidden_delta, inputs)

    def train(self):
        for inputs, output in zip(XOR.inputs, XOR.outputs):
            self.forward_propagation(inputs)
            self.back_propagation(inputs, output)
