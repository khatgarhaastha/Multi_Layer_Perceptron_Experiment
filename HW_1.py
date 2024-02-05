# Implementing Multi Layer Perceptron (MLP) with 1 hidden layer using numpy
import numpy as np
import pandas as pd

# Initalize parameters
def initalize_parameters(n_input, n_hidden, n_output):
    W1 = np.random.randn(n_input, n_hidden)*0.01 # weight matrix, 
    W2 = np.random.randn(n_hidden, n_output)*0.01 # weight matrix, avoiding vanishing gradient by multiplying with 0.01
    b1 = np.zeros((1, n_hidden)) # bias term, shape (1, n_hidden)
    b2 = np.zeros((1, n_output)) # bias term, shape (1, n_output)
    return W1, W2, b1, b2

# Activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivative of sigmoid because we need it for backpropagation
def sigmoid_derivative(x):
    return x*(1-x)

# Forward Propagation
def forward_propagation(X, W1, W2, b1, b2):
    Z1 = np.dot(X, W1) + b1 # linear transformation of input X
    A1 = sigmoid(Z1)  # activation of hidden layer
    Z2 = np.dot(A1, W2) + b2 # linear transformation of hidden layer
    A2 = sigmoid(Z2) # activation of output layer
    return Z1, A1, Z2, A2

# Compute loss
def compute_loss(Y, A2):
    m = Y.shape[0] # number of samples
    loss = -1/m * np.sum(Y*np.log(A2) + (1-Y)*np.log(1-A2)) # cross entropy loss
    return loss

# Backward Propagation
def backpropagation(X, Y, Z1, A1, Z2, A2, W1, W2):
    m = Y.shape[0] # number of samples
    dZ2 = A2 - Y # derivative of loss with respect to Z2
    dW2 = 1/m * np.dot(A1.T, dZ2) # derivative of loss with respect to W2
    db2 = 1/m * np.sum(dZ2, axis=0, keepdims=True) # derivative of loss with respect to b2
    dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(A1) # derivative of loss with respect to Z1
    dW1 = 1/m * np.dot(X.T, dZ1) # derivative of loss with respect to W1
    db1 = 1/m * np.sum(dZ1, axis=0, keepdims=True) # derivative of loss with respect to b1
    return dW1, dW2, db1, db2

# Update parameters
def update_parameters(W1, W2, b1, b2, dW1, dW2, db1, db2, learning_rate):
    W1 = W1 - learning_rate * dW1
    W2 = W2 - learning_rate * dW2
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2
    return W1, W2, b1, b2

# MLP model
def model(X,Y, n_hidden, num_iterations, learning_rate ):
    n_input = X.shape[1] # number of input features
    n_output = Y.shape[1] # number of output features

    W1, W2, b1, b2 = initalize_parameters(n_input, n_hidden, n_output) # initalize parameters

    for i in range(num_iterations):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, W2, b1, b2) 
        loss = compute_loss(Y, A2)
        dW1, dW2, db1, db2 = backpropagation(X, Y, Z1, A1, Z2, A2, W1, W2)
        W1, W2, b1, b2 = update_parameters(W1, W2, b1, b2, dW1, dW2, db1, db2, learning_rate)

        if i % 100 == 0:
            print(f"Loss after iteration {i}: {loss}")

    return W1, W2, b1, b2

# evaluate model
def predict(X, Y, W1, W2, b1, b2):
    Z1, A1, Z2, A2 = forward_propagation(X, W1, W2, b1, b2)
    predictions = np.round(A2) # rounding off the predictions
    accuracy = np.mean(predictions == Y) # accuracy of the model
    return accuracy 

# Function to run the experiment
def run_experiment(train, test, validate, n_hidden, num_iterations, learning_rate):

    # load data
    X_train, Y_train = load_data(train)
    #X_test, Y_test = load_data(test)
    X_validate, Y_validate = load_data(validate)

    # train model
    W1, W2, b1, b2 = model(X_train, Y_train, n_hidden, num_iterations, learning_rate)

    # evaluate model
    train_accuracy = predict(X_train, Y_train, W1, W2, b1, b2)
    validate_accuracy = predict(X_validate, Y_validate, W1, W2, b1, b2)
    #test_accuracy = predict(X_test, Y_test, W1, W2, b1, b2)

    print(f"Train accuracy: {train_accuracy}")
    print(f"Validation accuracy: {validate_accuracy}")
    #print(f"Test accuracy: {test_accuracy}")

# load data
def load_data(path):
    data = pd.read_csv(path)
    X = data[['x1', 'x2']].values # input features in the form of numpy array
    Y = data[['label']].values.reshape(-1,1) # output features in the form of numpy array
    return X, Y

# main function
if __name__ == "__main__":
    for i in range(1, 5):
        run_experiment('spiral_train.csv', 'spiral_test.csv', 'spiral_valid.csv', i, 1000, 0.02) # run the experiment

