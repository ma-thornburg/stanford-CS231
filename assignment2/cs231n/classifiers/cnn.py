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
    #print(H_prime2, W_prime2)
    
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






class ExpandedConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]*3 -> [FC -> RELU]*2 -> FC
  
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
    self.use_batchnorm = True
    self.conv_bn_params = [{'mode': 'train'} for i in range(0, 6)]
    self.affine_bn_params = [{'mode': 'train'} for i in range(0, 2)]
    self.filter_size = filter_size
    self.num_filters = filter_size
    
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
    #print(H_prime, W_prime)
    
    # Conv Layer
    prev_volume = input_dim[0] 
    prev_spatial_dim = (input_dim[1], input_dim[2])
    #print(prev_spatial_dim)
    for layer in range(1, 7):
        # Initialize Filter Weights. We will use padding to preserve spacial output.
        self.params['conv_W{layer}'.format(layer=layer)] = (
            np.random.normal(0, weight_scale, (num_filters, prev_volume, filter_size, filter_size))
        )
        self.params['conv_b{layer}'.format(layer=layer)] = (
            np.zeros(num_filters)
        )
        
        # Batchnorm params 
        self.params['conv_gamma' + str(layer)] = np.ones(num_filters)
        self.params['conv_beta' + str(layer)] = np.zeros(num_filters)
        
        if layer % 2 == 1:
            H_prime = 1 + (prev_spatial_dim[0] - 2) / 2
            W_prime = 1 + (prev_spatial_dim[1] - 2) / 2
        
        #print("HW ITER", H_prime, W_prime)
            
        # Update prev volume. 
        prev_volume = num_filters
        prev_spatial_dim = (H_prime, W_prime)
    
    
    # First Affine Layer
    #print(num_filters, H_prime, W_prime)
    self.params['affine_W1'] = np.random.normal(0, weight_scale, (num_filters * H_prime * W_prime, hidden_dim))
    self.params['affine_b1'] = np.zeros(hidden_dim)
    
    # First Layer Bartchnorm Params
    self.params['affine_gamma1'] = np.ones(hidden_dim)
    self.params['affine_beta1'] = np.zeros(hidden_dim)
    
    # Second Affine Layer
    self.params['affine_W2'] = np.random.normal(0, weight_scale, (hidden_dim, hidden_dim))
    self.params['affine_b2'] = np.zeros(hidden_dim)
    
    # Second Layer Batchnorm Params
    self.params['affine_gamma2'] = np.ones(hidden_dim)
    self.params['affine_beta2'] = np.zeros(hidden_dim)
    
    # Final Affine Layer
    self.params['affine_W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
    self.params['affine_b3'] = np.zeros(num_classes)
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
    reg = self.reg 
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = self.filter_size
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
    prev_output = X
    conv_caches = {}
    for layer in [1, 3, 5]:
        # First Conv output
        #print("Prev output shape", layer, prev_output.shape)
        conv_forward_output, conv_cache = conv_bn_relu_conv_bn_relu_pool_forward(
            prev_output, 
            self.params['conv_W{layer}'.format(layer=layer)], 
            self.params['conv_b{layer}'.format(layer=layer)],
            self.params['conv_gamma{layer}'.format(layer=layer)],
            self.params['conv_beta{layer}'.format(layer=layer)],
            
            self.params['conv_W{layer}'.format(layer=layer+1)], 
            self.params['conv_b{layer}'.format(layer=layer+1)],
            self.params['conv_gamma{layer}'.format(layer=layer+1)],
            self.params['conv_beta{layer}'.format(layer=layer+1)],
            
            conv_param.copy(),
            self.conv_bn_params[layer - 1],
            conv_param.copy(),
            self.conv_bn_params[layer],
            pool_param.copy(),
        )
        conv_caches[layer] = conv_cache
        
        prev_output = conv_forward_output
    
    # Affine 1 
    #print("conv output shape", conv_forward_output.shape)
    #print("affine_W shape", self.params['affine_W{layer}'.format(layer=1)].shape)
    #print("affine_b shape", self.params['affine_b{layer}'.format(layer=1)].shape)
    #print("affine gamma shape", self.params['affine_gamma{layer}'.format(layer=1)].shape)
    affine1_forward_output, affine1_cache = affine_relu_bn_forward(
        conv_forward_output, 
        self.params['affine_W{layer}'.format(layer=1)],
        self.params['affine_b{layer}'.format(layer=1)],
        self.params['affine_gamma{layer}'.format(layer=1)],
        self.params['affine_beta{layer}'.format(layer=1)],
        self.affine_bn_params[0],
    )
    
    # Affine BN 
    affine2_forward_output, affine2_cache = affine_relu_bn_forward(
        affine1_forward_output, 
        self.params['affine_W{layer}'.format(layer=2)],
        self.params['affine_b{layer}'.format(layer=2)],
        self.params['affine_gamma{layer}'.format(layer=2)],
        self.params['affine_beta{layer}'.format(layer=2)],
        self.affine_bn_params[1],
    )
    
    # Affine 2 
    scores, affine3_cache = affine_forward(
        affine2_forward_output, 
        self.params['affine_W{layer}'.format(layer=3)],
        self.params['affine_b{layer}'.format(layer=3)],
    )
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
    reg_loss = 0 
    # Conv Reg Loss
    for index in range(1, 7):
        reg_loss += np.sum(np.square(self.params['conv_W{layer}'.format(layer=index)]))
    for index in range(1, 4):
        reg_loss += np.sum(np.square(self.params['affine_W{layer}'.format(layer=index)]))
    reg_loss *= self.reg
    #print(reg_loss)
    #print(scores_loss)
    loss = reg_loss + scores_loss
    
    dloss = 1
    
    # loss = scores_loss + reg_loss
    dscores_loss = np.ones(scores_loss.shape) * dloss
    dreg_loss = np.ones(reg_loss.shape) * dloss
    
    # Reg Loss
    for index in range(1, 7):
        current_W = self.params['conv_W{layer}'.format(layer=index)]
        grads['conv_W{layer}'.format(layer=index)] = 2 * current_W * reg * dreg_loss
    for index in range(1, 4):
        current_W = self.params['affine_W{layer}'.format(layer=index)]
        grads['affine_W{layer}'.format(layer=index)] = 2 * current_W * reg * dreg_loss 
    
    # scores_loss, scores_local_grad = softmax_loss(scores, y)
    dscores = scores_local_grad * dscores_loss
    
    # Last Affine
    af_bp3 = affine_backward(dscores, affine3_cache)
    daffine2_forward_output = af_bp3[0]
    grads['affine_W3'] += af_bp3[1]
    grads['affine_b3'] = af_bp3[2]
    
    # Affine Layer 2
    af_bp2 = affine_relu_bn_backward(daffine2_forward_output, affine2_cache)
    daffine1_forward_output = af_bp2[0]
    grads['affine_W2'] += af_bp2[1]
    grads['affine_b2'] = af_bp2[2]
    grads['affine_gamma2'] = af_bp2[3]
    grads['affine_beta2'] = af_bp2[3]
    
    # Affine Layer 1
    af_bp1 = affine_relu_bn_backward(daffine1_forward_output, affine1_cache)
    prev_conv_forward_output = af_bp1[0]
    grads['affine_W1'] += af_bp1[1]
    grads['affine_b1'] = af_bp1[2]
    grads['affine_gamma1'] = af_bp1[3]
    grads['affine_beta1'] = af_bp1[3]
    
    for layer in [5, 3, 1]:
        conv_bp = conv_bn_relu_conv_bn_relu_pool_backward(prev_conv_forward_output, conv_caches[layer])
        prev_conv_forward_output = conv_bp[0] 
        grads['conv_W{layer}'.format(layer=layer)] += conv_bp[1]
        grads['conv_b{layer}'.format(layer=layer)] = conv_bp[2]
        grads['conv_gamma{layer}'.format(layer=layer)] = conv_bp[3] 
        grads['conv_beta{layer}'.format(layer=layer)] = conv_bp[4]
        grads['conv_W{layer}'.format(layer=layer+1)] += conv_bp[5] 
        grads['conv_b{layer}'.format(layer=layer+1)] = conv_bp[6]
        grads['conv_gamma{layer}'.format(layer=layer+1)] = conv_bp[7] 
        grads['conv_beta{layer}'.format(layer=layer+1)] = conv_bp[8]
        
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads

