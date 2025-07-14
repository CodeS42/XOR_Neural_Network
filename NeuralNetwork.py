import numpy as np

class NeuralNetwork:
    def __init__(self):
        hidden_weights = np.random.rand(2, 2)
        hidden_bias = np.random.rand(2)
        output_hidden_weights = np.random.rand(2)
        output_bias = np.random.rand(1)