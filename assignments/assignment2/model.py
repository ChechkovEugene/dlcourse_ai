import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.fc1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.fc2 = FullyConnectedLayer(hidden_layer_size, n_output)
    
    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        for name, param in self.params().items():
            param.grad = None
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        fc1_out = self.fc1.forward(X)
        fc2_out = self.fc2.forward(fc1_out)
        loss, out_grad = softmax_with_cross_entropy(fc2_out, y)
        
        fc2_back = self.fc2.backward(out_grad)
        dW2 = self.fc2.params()['W'].grad
        dB2 = self.fc2.params()['B'].grad
        
        fc2W_l, dW2_l = l2_regularization(self.fc2.params()['W'].value, self.reg)
        fc2B_l, dB2_l = l2_regularization(self.fc2.params()['B'].value, self.reg)

        fc1_back = self.fc1.backward(fc2_back)
        dW1 = self.fc1.params()['W'].grad
        dB1 = self.fc1.params()['B'].grad
        
        fc1W_l, dW1_l = l2_regularization(self.fc1.params()['W'].value, self.reg)
        fc1B_l, dB1_l = l2_regularization(self.fc1.params()['B'].value, self.reg)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        self.fc2.params()['W'].grad = dW2 + dW2_l
        self.fc2.params()['B'].grad = dB2 + dB2_l 
        self.fc1.params()['W'].grad = dW1 + dW1_l
        self.fc1.params()['B'].grad = dB1 + dB1_l

        return loss + fc2W_l + fc2B_l + fc1W_l + fc1B_l

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)

        fc1_out = self.fc1.forward(X)
        fc2_out = self.fc2.forward(fc1_out)
        pred = np.argmax(fc2_out, axis = 1)
        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params

        result = {'W1': self.fc1.params()['W'], 'B1': self.fc1.params()['B'], 
                  'W2': self.fc2.params()['W'], 'B2': self.fc2.params()['B']}

        return result
