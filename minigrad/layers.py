from builtins import range
import numpy as np

def affine_forward(x, w, b):
    """
    Compute the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_K)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biasses, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None

    x_reshaped = x.reshape(x.shape[0], -1)
    out = x_reshaped @ w + b

    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ..., d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    db = dout.sum(axis=0)
    x_reshaped = x.reshape(x.shape[0], -1)
    dx_reshaped = dout @ w.T
    dx = dx_reshaped.reshape(x.shape[0], *x.shape[1:])
    dw = x_reshaped.T @ dout
    return dx, dw, db

def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape of x
    - cache: x
    """
    out = None
    out = np.maximum(0, x)
    cache = x
    return out, cache

def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of retified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape.
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x.
    """
    dx, x = None, cache
    dx = dout * (x > 0)
    return dx

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    N = len(y)

    probs = np.exp(x - x.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)

    loss = -np.log(probs[range(N), y]).sum() / N

    probs[range(N), y] -= 1
    dx = probs/N


    return loss, dx

def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output,  as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape.
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None
    
    if mode == "train":
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
    elif mode == "test":
        out = x
    
    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache

def dropout_backward(dout, cache):
    """Backward pass for interted dropout.
    
    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        dx = dout * mask
    elif mode == "test":
        dx = dout
    return dx

def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization
    
    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time
    
    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:
    
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    
    Note that the batch normalization paper suggests a different test-time
    behavior: they comput sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation setp: the torch7 implementation
    of batch normalization also uses running averages/
    
    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift parameter of shape (D,)
    - bn_param: Dictionary with the following keys:
        - mode: 'train' or 'test'; required
        - eps: Constant for numeric stability
        - momentum: Constant for running mean / variance.
        - running_var: Array of shape (D,) giving running variance of features
        
    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))
    out, cache = None, None
    if mode == "train":
        mu = x.mean(axis=0)
        var = x.var(axis=0)
        std = np.sqrt(var + eps)
        x_hat = (x - mu) / std
        out = gamma * x_hat + beta
        shape = bn_param.get('shape', (N, D))
        axis = bn_param.get('axis', 0)
        cache = x, mu, var, std, gamma, x_hat, shape, axis
        
        if axis == 0:
            running_mean = momentum * running_mean + (1 - momentum) * mu
            running_var = momentum * running_var + (1 - momentum) * var
    elif mode == "test":
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
    
    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache

def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.
    
    Inputs:
    - dout: upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.
    
    Returns a tuple of:
    - dx: Gradient with respect to inpus x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """

    dx, dgamma, dbeta = None, None, None
    x, mu, var, std, gamma, x_hat, shape, axis = cache

    dbeta = dout.reshape(shape, order='F').sum(axis)
    dgamma = (dout * x_hat).reshape(shape, order='F').sum(axis)

    dx_hat = dout * gamma
    dstd = -np.sum(dx_hat * (x-mu), axis=0) / (std**2)
    dvar = 0.5 * dstd / std
    dx1 = dx_hat / std + 2 * (x-mu) * dvar / len(dout)
    dmu = -np.sum(dx1, axis=0)
    dx2 = dmu / len(dout)

    dx = dx1 + dx2
    return dx, dgamma, dbeta

def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization
    
    Inputs:
    - dout: upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.
    
    Returns a tuple of:
    - dx: Gradient with respect to inpus x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    x, mu, var, std, gamma, x_hat, shape, axis = cache
    S = lambda x: x.sum(axis=0)
    dbeta = dout.reshape(shape, order='F').sum(axis)
    dgamma = (dout * x_hat).reshape(shape, order='F').sum(axis)
    dx = dout * gamma / (len(dout)*std)
    dx = len(dout)*dx - S(dx*x_hat)*x_hat - S(dx)
    return dx, dgamma, dbeta