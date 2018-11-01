import numpy as np

def mean_square_loss(Y, Y_hat):
    ### Your code here (1-2 lines) ###
    return None

def cross_entropy_loss(Y, Y_hat):
    ### Your code here (1-2 lines) ###
    return None

# make sure our output is the same
np.random.seed(0)

# generate some random values of y and y^
y = np.random.rand(3)
y_hat = np.random.rand(3)

# run the mean_square_loss and cross_entropy_loss
print("MSE loss = ",mean_square_loss(y, y_hat))
print("Cross Entroy Error loss = ",cross_entropy_loss(y, y_hat))
