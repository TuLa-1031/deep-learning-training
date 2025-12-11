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
perform well; howeeer the defaults values of the other hyperparameters should
work well for a variety of different problems.

For effifiency, update rules may pẻom in-place updates, mutating w and
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