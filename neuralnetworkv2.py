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
        self.output_size = output_size

    # customize the learning rate of the neural network. Default is 0.12
    def set_learning_rate(learning_rate = 0.12):
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        ### Your code here (about 1 line) ##
        return 

    ### You can try to implement the following activation functions! ###
    def tanh(self, x):
        return x

    def relu(self, x):
        return x

    def leaky_relu(self, x):
        return x

    # for back propagation    
    def sigmoid_prime(self, x):
        return x

    def tanh_prime(self, x):
        return x

    def relu_prime(self, x):
        return x

    def leaky_relu_prime(self, x):
        return x

    # implement forward propagation    
    def forward(self, X):
        # get the weights from the python dictionary params
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2'] 

        ### Your code here (about 4 lines) ###
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

    # save the trined weights into a file
    def output(self):
        if self.trained:
            out = open('nn_weights.dat', 'wb')
            pickle.dump(self.params, out)
        else:
            print("Please train the network with data first!") 

    # load trained weights from file
    def input(self, weight_file):
        self.params = pickle.load(weight_file)
        self.trained = True

    # return the size of the neural network
    def getsize(self):
        return self.input_size, self.hidden_size, self.output_size,
