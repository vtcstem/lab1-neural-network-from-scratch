import numpy as np
import pickle

class neuralnetwork:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01, learning_rate = 0.12):
        self.params = {}
        np.random.seed(2)
        self.params['W1'] = weight_init_std * np.random.randn(hidden_size, input_size)
        self.params['b1'] = np.zeros((hidden_size, 1))
        self.params['W2'] = weight_init_std * np.random.randn(output_size, hidden_size)
        self.params['b2'] = np.zeros((output_size, 1))

        self.cost = []
        self.m = 0
        self.learning_rate = 0.12

        self.cache = {}
        self.trained = False

        self.input_size = input_size
        self.hidden_size = hidden_size
        self. output_size = output_size

    def set_learning_rate(learning_rate = 0.12):
        self.learning_rate = learning_rate

    # implement the sigmoid activation function
    def sigmoid(self, x):
        ### Your codes here (1-2 lines) ###
        return 

    # implement forward propagation
    def forward(self, X): 
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2'] 

        ### Your codes here (4 lines) ###
        Z1 = 
        A1 = 
        Z2 = 
        A2 = 

        self.cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}


    def predict(self, X):
        if self.trained:
            self.forward(X)
            prediction = (self.cache["A2"] > 0.5)
            print("Output is generated.")
            return prediction
        else:
            print("Please train the network first!")
            return None

    def output(self):
        if self.trained:
            out = open('nn_weights.dat', 'wb')
            pickle.dump(self.params, out)
        else:
            print("Please train the network with data first!") 

    def input(self, weight_file):
        self.params = pickle.load(weight_file)
        self.trained = True

    def getsize(self):
        return self.input_size, self.hidden_size, self.output_size,
