import numpy as np

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
    - w: A numpy array giving the current weights.
    - dw: A numpy array of the shape as w giving the gradient of the
      loss with respect to w.
    - config: A dictionary containing hyperparameter values such as learning
      rate, momentum, etc. If the update rule requires caching values over many
      iteraions, then config will also hold these caches values.

Returns:
    - next_w: The next point after the update.
    - config: The config dictionary to be passed to the next iteration of the
      update rule.

NOTE: for most udpate rules, the defaults learning rate will probably not
perform well; however the defaults values of the other hyperparameters should
work well for a variety of different problems.

For effifiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""

def sgd(w, dw, config = None):
    """
    Performs vanilla stachastic gradient descent.
    
    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config

def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None

    mu = config.get("momentum")
    lr = config.get("learning_rate")
    v = mu * v - lr * dw
    next_w = w + v

    config["velocity"] = v
    return next_w, config

def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving avrage of squared gradient
    values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving avarage of secound moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    next_w = None

    cache = config["decay_rate"]*config["cache"] + (1-config["decay_rate"])*dw**2
    next_w = w - config["learning_rate"]*dw/(np.sqrt(cache) + config["epsilon"])
    config["cache"]=cache
    return next_w, config

def adam(w, dw, config=None):
    """
    Ues the Adam update rule, which incorporates moving averages of both the gradient
    and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid divinding by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None
    """
    config["t"] += 1
    config["m"] = config["beta1"]*config["m"] + (1 - config["beta1"])*dw
    config["v"] = config["beta2"]*config["v"] + (1-config["beta2"])*dw**2
    config["learning_rate"] = config["learning_rate"] *                   \
                              np.sqrt(1-config["beta2"]**config["t"]) /   \
                              (1-config["beta1"]**config["t"])
    next_w = w - config["learning_rate"]*config["m"]/(np.sqrt(config["v"]) + config["epsilon"])
    """
    keys = ['learning_rate', 'beta1', 'beta2', 'epsilon', 'm', 'v', 't']
    lr, b1, b2, eps, m, v, t = (config.get(k) for k in keys)
    config['t'] = t = t + 1
    config['m'] = m = b1 * m + (1 - b1) * dw
    config['v'] = v = b2 * v + (1 - b2) * (dw**2)
    mt = m / (1 - b1**t)
    vt = v / (1 - b2**t)
    next_w = w - lr * mt / (np.sqrt(vt) + eps)

    return next_w, config

def adamw(w, dw, config):
    """
    Uses the AdamW update rule, which incorparates moving average of both the gradient
    and its square and a bias correctiont term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving averege of first moment of gradient.
    - beta2: Decay rate for moving average of secound moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)
    config.setdefault("weight_decay", 1e-2)

    next_w = None
    keys = ['learning_rate', 'beta1', 'beta2', 'epsilon', 'm', 'v', 't', 'weight_decay']
    lr, b1, b2, eps, m, v, t, weight_decay = (config.get(k) for k in keys)
    next_w = w - weight_decay*lr*w
    config['t'] = t = t + 1
    config['m'] = m = b1*m + (1-b1)*dw
    config['v'] = v = b2*v + (1-b2)*dw**2
    m_hat = m / (1 - b1**t)
    v_hat = v / (1 - b2**t)
    next_w = next_w - lr* m_hat / (np.sqrt(v_hat) + eps)
    return next_w, config