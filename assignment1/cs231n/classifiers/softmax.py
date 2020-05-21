import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = len(X)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  loss = 0
  # Compute Data Loss and Gradient
  for row_index, example in enumerate(X):
    
        # Compute Loss.
        correct_class = y[row_index]
        scores = np.dot(W.T, example.reshape(example.shape[0], 1))
        C = (-1) * np.max(scores, axis=0)
        scores_with_C = scores + C
        #print(scores.shape)
        correct_class_score = scores[correct_class]
        correct_class_score_with_C = correct_class_score + C
        scores_exp = np.exp(scores_with_C)
        scores_exp_summed = np.sum(scores_exp)
        correct_class_exp = np.exp(correct_class_score_with_C)
        softmax = np.true_divide(
            correct_class_exp,
            scores_exp_summed
        )
        softmax_logged = (-1) * np.log(softmax)
        
        # Update Loss.
        loss += softmax_logged
        
        # Compute Gradient 
        dloss = 1 
        
        # softmax_logged = (-1) * np.log(softmax)
        dsoftmax = dloss * (-1) * (np.true_divide(1, softmax)) 
        #print(softmax.shape, dsoftmax.shape)
        
        # softmax = np.true_divide(correct_class_exp, scores_exp_summed)
        dcorrect_class_exp = np.true_divide(1, scores_exp_summed) * dsoftmax
        dscores_exp_summed = (-1) * np.true_divide(correct_class_exp[0], scores_exp_summed**2) * dsoftmax[0]
        #print(correct_class_exp.shape, dcorrect_class_exp.shape)
        #print(scores_exp_summed.shape, dscores_exp_summed.shape)
        
        # correct_class_exp = np.exp(correct_class_score_with_C)
        dcorrect_class_score_with_C = np.exp(correct_class_score_with_C) * dcorrect_class_exp
        # print(correct_class_score_with_C.shape, dcorrect_class_score_with_C.shape)
        
        # scores_exp_summed = np.sum(scores_exp)
        dscores_exp = np.ones(scores_exp.shape) * dscores_exp_summed
        #print(scores_exp.shape, dscores_exp.shape)
        
        # scores_exp = np.exp(scores_with_C)
        dscores_with_C = np.exp(scores_with_C) * dscores_exp
        #print(scores_with_C.shape, dscores_with_C.shape)
        
        # correct_class_score_with_C = correct_class_score + C
        dcorrect_class_score = 1 * dcorrect_class_score_with_C
        dC = np.array([np.sum(1 * dcorrect_class_score_with_C)])
        #print(correct_class_score.shape, dcorrect_class_score.shape)
        #print(C.shape, dC.shape)
        
        # correct_class_score = scores[correct_class]
        dscores = np.zeros(scores.shape)
        dscores[correct_class] = 1 * dcorrect_class_score
        # print(scores.shape, dscores.shape)
        
        # scores_with_C = scores + C
        dscores += (np.ones(scores.shape) * dscores_with_C)
        dC += np.sum(dscores_with_C)
        #print(scores.shape, dscores.shape)
        #print(C.shape, dC.shape)
        
        # C = (-1) * np.max(scores, axis=0)
        dscores_temp = np.zeros(scores.shape)
        dscores_temp[np.argmax(scores)] = np.array([-1.0 * dC])
        dscores += dscores_temp

        # scores = np.dot(W.T, example.reshape(example.shape[0], 1))
        #print(dW.shape, example.shape, dscores.shape)
        dW += np.dot(
            example.T.reshape(len(example.T), 1), 
            dscores.T
        )
   
  # Average out loss.
  loss /= num_train

  # Average out gradients 
  dW /= num_train
    
  # Compute Regularization Loss
  loss += np.sum(np.square(W)) * reg
  dW += 2 * W * reg
            
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = len(X)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  loss = 0
  # Compute Data Loss and Gradient
    
  # Compute Loss.
  correct_class = y
  scores = np.dot(X, W)
  C = (-1) * np.amax(scores, axis=1)[:, np.newaxis]
  scores_with_C = scores + C
  correct_class_score = np.choose(y, scores.T)[:, np.newaxis]
  correct_class_score_with_C = correct_class_score + C
  scores_exp = np.exp(scores_with_C)
  scores_exp_summed = np.sum(scores_exp, axis=1)[:, np.newaxis]
  correct_class_exp = np.exp(correct_class_score_with_C)
  softmax = np.true_divide(
      correct_class_exp,
      scores_exp_summed
  )
  softmax_logged = (-1) * np.log(softmax)

  # Update Loss.
  loss += np.sum(softmax_logged)

  # Compute Gradient 
  dloss = np.true_divide(1, num_train)

  # softmax_logged = (-1) * np.log(softmax)
  dsoftmax = dloss * (-1) * (np.true_divide(1, softmax)) 
  #print(softmax.shape, dsoftmax.shape)

  # softmax = np.true_divide(correct_class_exp, scores_exp_summed)
  dcorrect_class_exp = np.true_divide(1, scores_exp_summed) * dsoftmax
  dscores_exp_summed = (-1) * np.true_divide(correct_class_exp, scores_exp_summed**2) * dsoftmax
  #print(correct_class_exp.shape, dcorrect_class_exp.shape)
  #print(scores_exp_summed.shape, dscores_exp_summed.shape)

  # correct_class_exp = np.exp(correct_class_score_with_C)
  dcorrect_class_score_with_C = np.exp(correct_class_score_with_C) * dcorrect_class_exp
  #print(correct_class_score_with_C.shape, dcorrect_class_score_with_C.shape)

  # scores_exp_summed = np.sum(scores_exp, axis=1)[:, np.newaxis]
  #print(dscores_exp_summed.shape)
  #print(scores_exp.shape)
  dscores_exp = np.ones(scores_exp.shape) * dscores_exp_summed
  #print(scores_exp.shape, dscores_exp.shape)

  # scores_exp = np.exp(scores_with_C)
  #print(scores_with_C.shape)
  dscores_with_C = np.exp(scores_with_C) * dscores_exp
  #print(scores_with_C.shape, dscores_with_C.shape)

  # correct_class_score_with_C = correct_class_score + C
  dcorrect_class_score = 1 * dcorrect_class_score_with_C
  #print(C.shape, correct_class_score.shape)
  dC = 1 * dcorrect_class_score_with_C
  #print(correct_class_score.shape, dcorrect_class_score.shape)
  #print(C.shape, dC.shape)

  # correct_class_score = np.choose(y, scores.T)[:, np.newaxis]
  dscores = np.zeros(scores.shape)
  #print(scores.shape, correct_class.shape, dcorrect_class_score.shape)
  dscores[np.arange(scores.shape[0]), correct_class] = 1 * dcorrect_class_score[:, 0]
  #print(scores.shape, dscores.shape)

  # scores_with_C = scores + C
  dscores += (np.ones(scores.shape) * dscores_with_C)
  dC += np.sum(dscores_with_C, axis=1)[:, np.newaxis]
  #print(scores.shape, dscores.shape)
  #print(C.shape, dC.shape)

  # C = (-1) * np.amax(scores, axis=1)[:, np.newaxis]
  dscores_temp = np.zeros(scores.shape)
  dscores_temp[np.arange(scores.shape[0]), np.argmax(scores, axis=1)] = -1.0 * dC[:, 0]
  #print(dscores_temp.shape)
  dscores += dscores_temp
  #print(scores.shape, dscores.shape)

  # scores = np.dot(X, W)
  
  #print(dW.shape, X.shape, dscores.shape)
  dW += np.dot(
      X.T, 
      dscores
  )
   
  # Average out loss.
  loss /= num_train
    
  # Average out gradients 
  #dW /= num_train
    
  # Compute Regularization Loss
  loss += np.sum(np.square(W)) * reg
  dW += 2 * W * reg

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

