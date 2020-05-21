import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    # Compute first layer outputs by applying W1, adding b1, and applying relu.
    fl_multiply = np.dot(X, W1)
    fl_multiply_and_add = fl_multiply + b1
    first_layer_scores = np.maximum(0, fl_multiply_and_add)
    
    # Compute second layer outputs by applying W2 and adding b2
    sl_multiply = np.dot(first_layer_scores, W2)
    sl_scores = sl_multiply + b2
    
    # Update scores
    scores = sl_scores
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
    pass
    # Compute Data Loss
    f_y = np.choose(y, scores.T)
    exp_scores = np.exp(scores)
    sum_exp_scores = np.sum(exp_scores, axis=1)
    log_sum = np.log(sum_exp_scores)
    data_loss = (-1 * f_y) + log_sum
    mean_data_loss = (data_loss).mean()
    
    # Add L2 Regularization Loss
    W1_squared = np.square(W1)
    reg_W1 = W1_squared.sum()
    W2_squared = np.square(W2)
    reg_W2 = W2_squared.sum()
    reg_loss = (reg_W1 + reg_W2) * 0.5
    reg_loss_with_strength = reg_loss * reg
    loss = mean_data_loss + reg_loss_with_strength
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    dloss = 1
    
    # loss = mean_data_loss + reg_loss_with_strength
    dreg_loss_with_strength = dloss * 1
    dmean_data_loss = dloss * 1
    
    # reg_loss_with_strength = reg_loss * reg
    dreg_loss = reg * dreg_loss_with_strength
    dreg = reg_loss * dreg_loss_with_strength
    
    # reg_loss = (reg_W1 + reg_W2) * 0.5
    dreg_W1 = .5 * dreg_loss
    dreg_W2 = .5 * dreg_loss
    
    # reg_W2 = W2_squared.sum()
    dW2_squared = np.ones(W2_squared.shape) * dreg_W2
    
    # W2_squared = np.square(W2)
    dW2 = dW2_squared * 2 * W2
    
    # reg_W1 = W1_squared.sum()
    dW1_squared = np.ones(W1_squared.shape) * dreg_W1 
    
    # W1_squared = np.square(W1)
    dW1 = 2 * W1 * dW1_squared
    
    # mean_data_loss = (data_loss).mean()
    ddata_loss = np.true_divide(1, len(data_loss)) * np.ones(data_loss.shape) * dmean_data_loss 
    
    # data_loss = (-1 * f_y) + log_sum
    df_y = -1 * ddata_loss
    dlog_sum = 1 * ddata_loss
    
    # log_sum = np.log(sum_exp_scores)
    dsum_exp_scores = np.true_divide(1, sum_exp_scores) * dlog_sum
    
    # sum_exp_scores = np.sum(exp_scores, axis=1)
    dexp_scores = np.ones(exp_scores.shape) * dsum_exp_scores[:, np.newaxis]
    
    # exp_scores = np.exp(scores)
    dscores = np.exp(scores) * dexp_scores
    
    # f_y = np.choose(y, scores.T)
    _intermediate = np.zeros(scores.shape)
    _intermediate[np.arange(scores.shape[0]), y] = 1.0
    dscores += _intermediate * df_y[:, np.newaxis]
    
    # sl_scores = sl_multiply + b2
    dsl_multiply = 1 * dscores
    db2 = 1 * np.sum(dscores, axis=0)
    
    # sl_multiply = np.dot(first_layer_scores, W2)
    dW2 += np.dot(first_layer_scores.T, dsl_multiply)
    dfirst_layer_scores = np.dot(dsl_multiply,  W2.T)
    
    # first_layer_scores = np.maximum(0, fl_multiply_and_add)
    dfl_multiply_and_add = (fl_multiply_and_add > 0).astype(int) * dfirst_layer_scores
    
    # fl_multiply_and_add = fl_multiply + b1
    dfl_multiply = 1 * dfl_multiply_and_add
    db1 = 1 * np.sum(dfl_multiply_and_add, axis=0)
    
    # fl_multiply = np.dot(X, W1)
    dW1 += np.dot(X.T, dfl_multiply)
    
    # Update gradients
    grads['W1'] = dW1
    grads['W2'] = dW2
    grads['b1'] = db1
    grads['b2'] = db2
    
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      rand_indices = np.random.choice(num_train, batch_size, replace=False)
      X_batch = X[rand_indices, :]
      y_batch = y[rand_indices]
      pass
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      dW1, dW2, db1, db2 = grads['W1'], grads['W2'], grads['b1'], grads['b2']
      self.params['W1'] += (-1) * learning_rate * grads['W1']
      self.params['W2'] += (-1) * learning_rate * grads['W2']
      self.params['b1'] += (-1) * learning_rate * grads['b1']
      self.params['b2'] += (-1) * learning_rate * grads['b2']
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    y_pred = np.argmax(self.loss(X), axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


