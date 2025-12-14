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
    
    
class FullyConnectedNet(object):
    """Class for a multi-layer full connected neural network.
    
    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be
    
    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax
    
    where batch/layer normalizaion and dropout are optional and the {...} block is
    repeated L - 1 times.
    
    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
            self,
            hidden_dims,
            input_dim = 3*32*32,
            num_classes=10,
            dropout_keep_ratio=1,
            normalization=None,
            reg=0.0,
            weight_scale=1e-2,
            dtype=np.float32,
            seed=None,
    ):
        """Initizalize a new FullyConnectedNet.
        
        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the output.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the netword should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
           are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float332 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layer deteriminstic so we can grdient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        for l, (i, j) in enumerate(zip([input_dim, *hidden_dims], [*hidden_dims, num_classes])):
            self.params[f"W{l+1}"] = np.random.randn(i, j) * weight_scale
            self.params[f'b{l+1}'] = np.zeros(j)

            #if self.normalization and l < self.num_layers - 1:
            #    self.params[f'gamma{l+1}'] = np.ones(j)
            #    self.params[f'beta{l+1}'] =  np.zeros(j)

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
        
    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of lables, of shape (N,). y[i] gives the label for X[i].
        
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
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"
    
        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = ModuleNotFoundError
        scores = None

        cache = {}
        for l in range(self.num_layers):
            keys = [f'W{l+1}', f'b{l+1}', f'gamma{l+1}', f'beta{l+1}']
            w, b, gamma, beta = (self.params.get(k, None) for k in keys)
            bn = self.bn_params[l] if gamma is not None else None
            do = self.dropout_param if self.use_dropout else None

            X, cache[l] = generic_forward(X, w, b, do,gamma, beta, bn, l==self.num_layers-1)
        scores = X

        if mode == "test":
            return scores
        
        loss, grads = 0.0, {}
        loss, dout = softmax_loss(scores, y)
        loss += 0.5 * self.reg * np.sum([np.sum(W**2) for k, W in self.params.items() if 'W' in k])
        for l in reversed(range(self.num_layers)):
            dout, dW, db, dgamma, dbeta = generic_backward(dout, cache[l])
            grads[f'W{l+1}'] = dW + self.reg * self.params[f'W{l+1}']
            grads[f'b{l+1}'] = db
            if dgamma is not None and l < self.num_layers-1:
                grads[f'gamma{l+1}'] = dgamma
                grads[f'beta{l+1}'] = dbeta
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