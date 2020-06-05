import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    self.num_classes = num_classes
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    # Weights should be of dimension (D, H) (N, D) (D, C)
    W1 = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
    b1 = np.zeros(hidden_dim)
    W2 = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
    b2 = np.zeros(num_classes)
    
    self.params = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
    }
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    N = X.shape[0]
    D = np.prod(X.shape[1:])
    C = self.num_classes
    
    W1, W2 = self.params['W1'], self.params['W2']
    b1, b2 = self.params['b1'], self.params['b2']
    reg = self.reg
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    first_affine_out, first_affine_cache = affine_forward(X, W1, b1)
    first_relu_out, first_relu_cache = relu_forward(first_affine_out)
    scores, second_affine_cache = affine_forward(first_relu_out, W2, b2)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    scores_loss, scores_local_grad = softmax_loss(scores, y)
    reg_loss = reg * 0.5 * (
        (np.sum(np.square(W1))) + 
         (np.sum(np.square(W2)))
    )
    #print(reg_loss, scores_loss)
    loss = scores_loss + reg_loss
           
    dloss = 1
    
    # loss = scores_loss + reg_loss
    dscores_loss = dloss * (1)
    dreg_loss = dloss * (1)
    
    #  reg_loss = reg * 0.5 * (
    #    (np.sum(np.square(W1))) + 
    #     (np.sum(np.square(W2))) 
    # )
    dW1 = dreg_loss * (reg * W1)
    dW2 = dreg_loss * (reg * W2)
   
    # scores_loss, scores_local_grad = softmax_loss(scores, y)
    dscores = scores_local_grad * dscores_loss
    
    # scores, second_affine_cache = affine_forward(first_relu_out, W2, b2)
    scores_bp = affine_backward(dscores, second_affine_cache) 
    dfirst_relu_out = scores_bp[0]
    dW2 += scores_bp[1]
    db2 = scores_bp[2]
    
    # first_relu_out, first_relu_cache = relu_forward(first_affine_out)
    dfirst_affine_out = relu_backward(dfirst_relu_out, first_relu_cache)
    
    # first_affine_out, first_affine_cache = affine_forward(X, W1, b1)
    first_affine_out_bp = affine_backward(dfirst_affine_out, first_affine_cache)
    dW1 += first_affine_out_bp[1]
    db1 = first_affine_out_bp[2]
    
    grads['W1'] = dW1
    grads['W2'] = dW2
    grads['b1'] = db1
    grads['b2'] = db2
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    #print(loss)
    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    # Initialize Paramaters for layers.
    prev_input_dim = input_dim
    for index, hidden_dim in enumerate(hidden_dims):
        self.params['W' + str(index + 1)] = (
            np.random.normal(0, weight_scale, (prev_input_dim, hidden_dim))
        )
        self.params['b' + str(index + 1)] = (
            np.zeros(hidden_dim)
        )
        prev_input_dim = hidden_dim
    
    # Initialize Paramaters for output layer. 
    self.params['W' + str(self.num_layers)] = np.random.normal(0, weight_scale, (prev_input_dim, num_classes))
    self.params['b' + str(self.num_layers)] = np.zeros(num_classes)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################
    intermediate_X = X
    network_caches = {}
    for index in range(1, self.num_layers):
        # Lookup the layers paramaters.
        W, b = self.params['W' + str(index)], self.params['b' + str(index)]
        
        # Compute this layers affine relu output and forward pass caches. 
        out, cache = affine_relu_forward(intermediate_X, W, b)
        
        # Store this layers cache. 
        network_caches[index] = cache
        
        # Update intermediate_X
        intermediate_X = out
     
    # Produce score for last layer. 
    W, b = self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)]
    scores, cache = affine_forward(intermediate_X, W, b)
    network_caches[self.num_layers] = cache
        
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    scores_loss, scores_local_grad = softmax_loss(scores, y)
    reg_loss = 0 
    for index in range(1, self.num_layers + 1):
        reg_loss += np.sum(np.square(self.params['W' + str(index)]))
    reg_loss *= (0.5 * self.reg)
    loss = reg_loss + scores_loss
    
    dloss = 1 
    dreg_loss = dloss * 1
    dscores_loss = dloss * 1
    
    # reg_loss = 0 
    # for index in range(1, self.num_layers):
    #    reg_loss += np.sum(np.square(self.params['W' + str(index)]))
    # reg_loss += np.sum(np.square('Wfinal'))
    # reg_loss *= (0.5 * reg)
    for index in range(1, self.num_layers + 1):
        grads['W' + str(index)] = dreg_loss * self.reg * self.params['W' + str(index)]
        
    # scores_loss, scores_local_grad = softmax_loss(scores, y)
    dscores = dscores_loss * scores_local_grad
    
    # W, b = self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)]
    #scores, cache = affine_forward(intermediate_X, W, b)
    bp = affine_backward(dscores, network_caches[self.num_layers])
    grads['W' + str(self.num_layers)] += bp[1]
    grads['b' + str(self.num_layers)] = bp[2]
    
    
    #intermediate_X = X
    #network_caches = {}
    #for index in range(1, self.num_layers):
        ## Lookup the layers paramaters.
        # W, b = self.params['W' + str(index)], self.params['b' + str(index)]
        ## Compute this layers affine relu output and forward pass caches. 
        #out, cache = affine_relu_forward(intermediate_X, W, b)
        ## Store this layers cache. 
        # network_caches[index] = cache
        ## Update intermediate_X
        #intermediate_X = out
    prev_dout = bp[0]
    for index in reversed(range(1, self.num_layers)):
        bp = affine_relu_backward(prev_dout, network_caches[index])
        grads['W' + str(index)] += bp[1]
        grads['b' + str(index)] = bp[2]
        prev_dout = bp[0]
        
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
