from layers import *

def affine_relu_forward(x, w, b):
    """
    Covenience layer that performs an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

def generic_forward(x, w, b, last=None):
    """Convenience layer that performs an affine transform, a batch/layer normalization

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - last: Indicates wether to perform just affine forward

    Return a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to backward pass
    """
    # Initialize optinal cache to None
    relu_cache = None
    out, fc_cache = affine_forward(x, w, b)
    if not last:
        out, relu_cache = relu_forward(out)
    cache = fc_cache, relu_cache
    return out, cache

def generic_backward(dout, cache):
    """
    Backward pass for the generic convenience layer
    """
    fc_cache, relu_cache = cache
    if relu_cache is not None:
        dout = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(dout, fc_cache)
    return dx, dw, db
