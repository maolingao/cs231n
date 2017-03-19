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
  num_train, dim = X.shape
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # pass
  for i in xrange(num_train) :
    score = X[i,:].dot(W)                               # scores for all classes
    score_exp = np.exp(score)                           # some preparation computing
    sum_score_exp = np.sum(score_exp)
    sum_score_exp_sqrt = sum_score_exp**2
    loss += -np.log( score_exp[y[i]] / sum_score_exp )  # softmax function
    
    for j in xrange(num_classes) :
        factor = 1 / ( sum_score_exp + 1e-8)
        if ( j!=y[i] ) :
            dW[:,j] += factor  * score_exp[j] * X[i,:].T
        else :
            dW[:,j] += -factor * (sum_score_exp - score_exp[y[i]] ) * X[i,:].T
        
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW   /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * ( np.sum(W * W) - np.sum(W[-1,:] * W[-1,:]) ) ## bias shouldn't go into the regularization.
                                                                      ## the l2 regularization just prefers the small and separat weight along all dim
                                                                      ## which shouldn't affect the bias.
  dW_reg = reg * W   ## weighting and bias should be both updated.
  dW_reg[-1, 0] = 0  ## but the update of the bias should only come from the data loss.
  dW += dW_reg;
  
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
  num_train, dim = X.shape
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # pass
  # data loss
  score_mtx = X.dot(W)
  score_mtx_exp = np.exp(score_mtx)
  # print score_mtx_exp[1, 1:10]
  sum_score_exp_mtx_per_row = np.sum(score_mtx_exp, axis=1)
 #  print 'sum_score_exp_mtx_per_row[0] = %f' % (sum_score_exp_mtx_per_row[0])
  score_mtx_exp_correct_class = score_mtx_exp[xrange(num_train), y]
  factor = 1 / (sum_score_exp_mtx_per_row + 1e-8)
  loss_per_image = -np.log(score_mtx_exp_correct_class * factor + 1e-8)  # softmax function
 #  print 'loss_per_image[0] = %f' % (loss_per_image[0])
  loss = np.sum(loss_per_image)
  
  # regularization
  mask = score_mtx_exp
  mask[xrange(num_train), y] = - ( sum_score_exp_mtx_per_row - score_mtx_exp_correct_class )
  mask = (mask.T * factor).T
  dW = X.T.dot(mask)
  
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW   /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * ( np.sum(W * W) - np.sum(W[-1,:] * W[-1,:]) ) ## bias shouldn't go into the regularization.
                                                                      ## the l2 regularization just prefers the small and separat weight along all dim
                                                                      ## which shouldn't affect the bias.
  dW_reg = reg * W   ## weighting and bias should be both updated.
  dW_reg[-1, 0] = 0  ## but the update of the bias should only come from the data loss.
  dW += dW_reg;
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

