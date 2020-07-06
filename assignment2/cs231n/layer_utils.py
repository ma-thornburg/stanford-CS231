from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

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


def affine_relu_bn_forward(x, w, b, gamma, beta, bn_param):
    """
    Forward pass for the affine-bn-relu convenience layer
    """
    fc_out, fc_cache = affine_forward(x, w, b)
    bn_out, bn_cache = batchnorm_forward(fc_out, gamma, beta, bn_param)
    out, relu_cache = relu_forward(bn_out)
    cache = (fc_cache, bn_cache, relu_cache)
    return (out, cache)

def affine_relu_bn_backward(dout, cache):
    """
    Forward pass for the affine-bn-relu convenience layer
    """
    fc_cache, bn_cache, relu_cache = cache
    dbn = relu_backward(dout, relu_cache)
    dfcout, dgamma, dbeta = batchnorm_backward(dbn, bn_cache)
    dx, dw, db = affine_backward(dfcout, fc_cache)
    return (dx, dw, db, dgamma, dbeta)

def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db

def conv_bn_relu_conv_bn_relu_pool_forward(x, w1, b1, gamma1, beta1, w2, b2, gamma2, beta2, conv_param1, bn_param1, conv_param2, bn_param2, pool_param1):
    """
    Forward pass for conv-bn-relu-conv-bn-relu-pool layer.
    """
    first_conv_output, first_conv_cache = conv_forward_fast(x, w1, b1, conv_param1)
    first_bn_output, first_bn_cache = spatial_batchnorm_forward(first_conv_output, gamma1, beta1, bn_param1)
    first_relu_output, first_relu_cache = relu_forward(first_bn_output)
    #print("FP1", first_conv_output.shape)
    
    second_conv_output, second_conv_cache = conv_forward_fast(first_relu_output, w2, b2, conv_param2)
    second_bn_output, second_bn_cache = spatial_batchnorm_forward(second_conv_output, gamma2, beta2, bn_param2)
    second_relu_output, second_relu_cache = relu_forward(second_bn_output)
    #print("FP2", second_conv_output.shape)
    
    out, pool_cache = max_pool_forward_fast(second_relu_output, pool_param1)
    #print("out shape", out.shape)
    
    cache = (
        first_conv_cache, first_bn_cache, first_relu_cache, second_conv_cache, 
        second_bn_cache, second_relu_cache, pool_cache
    )
    
    return out, cache

def conv_bn_relu_conv_bn_relu_pool_backward(dout, cache):
    """
    Backward pass for conv-bn-relu-conv-bn-relu-pool layer. 
    """
    first_conv_cache, first_bn_cache, first_relu_cache, second_conv_cache, second_bn_cache, second_relu_cache, pool_cache = cache
    
    pool_bp = max_pool_backward_fast(dout, pool_cache)
    second_relu_bp = relu_backward(pool_bp, second_relu_cache)
    second_bn_bp, dgamma2, dbeta2, = spatial_batchnorm_backward(second_relu_bp, second_bn_cache)
    second_conv_bp, dw2, db2 = conv_backward_fast(second_bn_bp, second_conv_cache)
    first_relu_bp = relu_backward(second_conv_bp, first_relu_cache)
    first_bn_bp, dgamma1, dbeta1 = spatial_batchnorm_backward(first_relu_bp, first_bn_cache)
    dx, dw1, db1 = conv_backward_fast(first_bn_bp, first_conv_cache)
    
    return dx, dw1, db1, dgamma1, dbeta1, dw2, db2, dgamma2, dbeta2
    
    
                          