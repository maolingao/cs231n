import numpy as np
from random import shuffle

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

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
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # dW[:,j]    += X[i].T
        # dW[:,y[i]] -= X[i].T
        dW[:,j]    += X.T[:,i]
        dW[:,y[i]] -= X.T[:,i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW   /= num_train
  
  # Add regularization to the loss.
  # loss += 0.5 * reg * np.sum(W * W)
  loss += 0.5 * reg * ( np.sum(W * W) - np.sum(W[-1,:] * W[-1,:]) ) ## bias shouldn't go into the regularization.
                                                                      ## the l2 regularization just prefers the small and separat weight along all dim
                                                                      ## which shouldn't affect the bias.
  dW_reg = reg * W   ## weighting and bias should be both updated.
  dW_reg[-1, 0] = 0  ## but the update of the bias should only come from the data loss.
  dW += dW_reg;
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # pass
  score_matrix = X.dot(W)
  correct_class_score_vector = score_matrix[range(num_train), y]
  margin_matrix = np.clip(score_matrix.T - correct_class_score_vector + 1, 0, float("inf"))  # 1 added to much per training image
  margin_matrix = margin_matrix.T
  
  loss = np.sum(margin_matrix) - num_train
  
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * ( np.sum(W * W) - np.sum(W[-1,:] * W[-1,:]) ) ## bias shouldn't go into the regularization.
                                                                      ## the l2 regularization just prefers the small and separat weight along all dim
                                                                      ## which shouldn't affect the bias.
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # pass
  mask = np.zeros(margin_matrix.shape)
  mask[margin_matrix > 0] = 1;
  row_sum = np.sum(mask, axis=1) ## note: should eliminate the score=1 counted for the correct class
  mask[range(num_train), y] -= row_sum
  dW = mask.T.dot(X) / num_train ## gradient of the data loss part
  dW = dW.T
  
  dW_reg = reg * W   ## weighting and bias should be both updated.
  dW_reg[-1, 0] = 0  ## but the update of the bias should only come from the data loss.
  dW += dW_reg;
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
