from NeuralNetwork import NeuralNetwork

def main():
    epoch = 0
    
    neural_network = NeuralNetwork()
    while epoch < 1000:
        neural_network.train()

if __name__ == "__main__":
    main()