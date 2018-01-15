import numpy as np
from random import shuffle


def marginalize(x):
    """Set a nonnegative input to 0."""
    if x > 0:
        return x
    else:
        return 0

def is_past_margin(x):
    """Compute whether an input was greater than 0."""
    if x > 0:
        return 1
    else:
        return 0

marginalizev = np.vectorize(marginalize, otypes=[np.float64])
is_past_marginv = np.vectorize(is_past_margin)

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    # initialize the gradient as zero
    dW = np.zeros(W.shape)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                loss += margin
                dW[:, y[i]] += - X[i]
                dW[:, j] += X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Since we are averaging the loss, we also need to average the gradients.
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)

    # Add the regularization gradient. Since this is done outside of the
    # averaging, we don't need to average here.
    dW += .5 * reg * (2 * W)
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.

    TODO(mathorn92): Despite being vectorized, this function
    still runs slower than naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    num_classes = W.shape[1]
    num_train = X.shape[0]

    scores = X.dot(W)
    correct_class_scores = np.choose(y, scores.T).reshape(len(y), 1)

    # First calculate all margins.
    margins = scores - correct_class_scores + 1

    # Then set the first correct classes margin to 0.
    margins[np.arange(margins.shape[0]), y] = 0

    # Then set all negative margins to 0.
    margins = marginalizev(margins)

    # Then add all the margins to the loss.
    loss += np.sum(margins)

    # Increment incorrect class dW.
    is_past_margins_vect = is_past_marginv(margins)
    dW += X.transpose().dot(is_past_margins_vect)

    # Increment correct class dW. Do this by subtracting X[i] times
    # the number of incorrect classes that past the margin from the
    # correct class dW
    is_past_margin_weights = np.sum(is_past_margins_vect, axis=1)
    is_past_margin_weights = is_past_margin_weights.reshape(len(is_past_margin_weights), 1)

    dWT = dW.transpose()
    # This line is slow.
    np.add.at(dWT, (y, slice(None)), -1 * (X * is_past_margin_weights))

    dW = dWT.transpose()

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Since we are averaging the loss, we also need to average the gradients.
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)

    # Add the regularization gradient. Since this is done outside of the
    # averaging, we don't need to average here.
    dW += .5 * reg * (2 * W)
    return loss, dW
