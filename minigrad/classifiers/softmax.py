from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss Function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minbatches
    of N examples.
    
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N, ) containing training labels; y[i]
    = c means that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    num_dims = W.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        #compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum() #normalize
        logp = np.log(p)

        loss -= logp[y[i]] 
        p[y[i]] -= 1
        dW += np.outer(X[i], p)

    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + 2 * reg*W


    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and the gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_train = X.shape[0]
    scores = X @ W

    probs = np.exp(scores - scores.max())
    probs /= probs.sum(axis=1, keepdims=True)

    loss -= np.log(probs[range(num_train), y]).sum()
    loss = loss / num_train + reg * np.sum(W * W)

    probs[range(num_train), y] -= 1
    dW = X.T @ probs / num_train + 2 * reg * W


    return loss, dW