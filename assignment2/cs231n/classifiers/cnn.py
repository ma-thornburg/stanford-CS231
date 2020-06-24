import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    # Conv Setup
    # H' = 1 + (H + 2 * pad - HH) / stride 
    # W' = 1 + (W + 2 * pad - WW) / stride 
    stride = 1
    pad = (filter_size - 1) / 2
    H_prime = 1 + (input_dim[1] + 2 * pad - filter_size) / stride
    W_prime = 1 + (input_dim[2] + 2 * pad - filter_size) / stride
    H_prime2 = 1 + (H_prime - 2) / 2
    W_prime2 = 1 + (W_prime - 2) / 2
    print(H_prime2, W_prime2)
    
    # Conv Layer
    self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, input_dim[0], filter_size, filter_size))
    self.params['b1'] = np.zeros(num_filters)
    
    # First Affine Layer
    self.params['W2'] = np.random.normal(0, weight_scale, (num_filters * H_prime2 * W_prime2, hidden_dim))
    self.params['b2'] = np.zeros(hidden_dim)
    
    # Second Affine Layer
    self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    reg = self.reg 
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    # Conv
    conv_forward_output, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    
    # Affine 1 
    affine1_forward_output, affine1_cache = affine_relu_forward(conv_forward_output, W2, b2)
    
    # Affine 2 
    scores, affine2_cache = affine_forward(affine1_forward_output, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    # Scores Loss / Grad
    scores_loss, scores_local_grad = softmax_loss(scores, y)
    
    # Reg Loss - Skip biases in reg loss
    reg_loss = (
        np.sum(W1 * W1) + 
        np.sum(W2 * W2) +
        np.sum(W3 * W3) 
    ) * reg
    loss = scores_loss + reg_loss
    
    dloss = 1
    
    # loss = scores_loss + reg_loss
    dscores_loss = np.ones(scores_loss.shape) * dloss
    dreg_loss = np.ones(reg_loss.shape) * dloss
    
    # reg_loss = (
    #    np.sum(W1 * W1) + 
    #    np.sum(W2 * W2) +
    #    np.sum(W3 * W3) + 
    # ) * reg
    grads['W1'] = 2 * W1 * reg * dreg_loss 
    grads['W2'] = 2 * W2 * reg * dreg_loss
    grads['W3'] = 2 * W3 * reg * dreg_loss
    
    # scores_loss, scores_local_grad = softmax_loss(scores, y)
    dscores = scores_local_grad * dscores_loss
    
    # scores, affine2_cache = affine_forward(affine1_forward_output, W3, b3)
    bp3 = affine_backward(dscores, affine2_cache)
    daffine1_forward_output = bp3[0]
    grads['W3'] += bp3[1]
    grads['b3'] = bp3[2]
    
    # affine1_forward_output, affine1_cache = affine_relu_forward(conv_forward_output, W2, b2)
    bp2 = affine_relu_backward(daffine1_forward_output, affine1_cache)
    dconv_forward_output = bp2[0]
    grads['W2'] += bp2[1]
    grads['b2'] = bp2[2]
    
    # conv_forward_output, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    bp1 = conv_relu_pool_backward(dconv_forward_output, conv_cache)
    grads['W1'] += bp1[1]
    grads['b1'] = bp1[2]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
