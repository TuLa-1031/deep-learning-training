from builtins import range
from builtins import object
import os
import numpy as np

from layers import *
from layer_utils import *

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer desgin. We assum an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; istead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """


    def __init__(
            self,
            input_dim= 3 * 32 * 32,
            hidden_dim = 100,
            num_classes = 10,
            weight_scale = 1e-3,
            reg = 0.0,
    ):
            """
            Initialize a new network.

            Inputs:
            - input_dim: An integer giving the size of the input
            - hidden_dim: An integer giving the size of the hidden layer
            - num_classes: An integer giving the number of class to classify
            - weight_scale: Scalar giving the standard deviation for random
              initialization of the weights.
            - reg: Scalar giving L2 regularization strength.
            """
            self.params = {
                  "W1": np.random.normal(size=(input_dim, hidden_dim)) * weight_scale,
                  "W2": np.random.normal(size=(hidden_dim, num_classes)) * weight_scale,
                  "b1": np.zeros(shape=(hidden_dim,)),
                  "b2": np.zeros(shape=(num_classes,)),
            }
            self.reg = reg

    def loss(self, X, y=None):
        """
        Compute the loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then ren a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
        scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
        names to gradients of the loss with respect to those parameters.
        """
        scores = None
        W1, W2, b1, b2 = self.params.values()
        hidden, cache1 = affine_relu_forward(X, W1, b1)
        scores, cache2 = affine_forward(hidden, W2, b2)
        
        if y is None:
            return scores
        loss, grads = 0, {}
        loss, dloss = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2))
        dout2, dW2, db2 = affine_backward(dloss, cache2)
        dout1, dW1, db1 = affine_relu_backward(dout2, cache1)

        dW1 += self.reg*W1
        dW2 += self.reg*W2

        grads = {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
        }
        return loss, grads
    
    def save(self, fname):
        """Save model parameters."""
        fpath = os.path.join(os.path.dirname(__file__), "../saved", fname)
        params = self.params
        np.save(fpath, params)
        print(fname, "saved.")

    def load(self, fname):
        """Load model parameters."""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        if not os.path.exists(fpath):
            print(fname, "not available.")
            return False
        else:
            params = np.load(fpath, allow_pickle=True).item()
            self.params = params
            print(fname, "loaded.")
            return True