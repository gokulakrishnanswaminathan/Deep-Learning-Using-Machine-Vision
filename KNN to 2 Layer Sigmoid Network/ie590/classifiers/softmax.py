'''
This module contains three functions (naive, partially-vectorized and fully-vectorized)
computing the loss and the gradients. 
'''

import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with two loops)

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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss variable and the gradient in dW. If you are not    #
    # careful here, it is easy to run into numeric instability. Don't forget    #
    # the regularization both in loss and in the gradient computations!         #
    #############################################################################
    #                          START OF YOUR CODE                               #
    #############################################################################
    scores = X.dot(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):
        a = scores[i] - np.max(scores[i])
        softmax = np.exp(a)/np.sum(np.exp(a))
        loss += -np.log(softmax[y[i]])
        for j in range(num_classes):
            dW[:,j] += X[i]*softmax[j]
        dW[:,y[i]] -= X[i]
    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W*W)
    dW += reg * 2 * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

def softmax_loss_partially_vectorized(W, X, y, reg):
    """
    Softmax loss function, partially vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using one explicit loop.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # the regularization both in loss and in the gradient computations!         #
    #############################################################################
    #                          START OF YOUR CODE                               #
    #############################################################################
    sW=np.zeros_like(X.dot(W))
    scores = X.dot(W)
    num_train=X.shape[0]
    num_classes=W.shape[1]
    for i in range(num_train):
        a = scores[i] - np.max(scores[i])
        softmax = np.exp(a)/np.sum(np.exp(a))
        loss += -np.log(softmax[y[i]])
        softmax[y[i]] -= 1
        dW += X[i].reshape(X.shape[1],1).dot(softmax.reshape(1,dW.shape[1]))
    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W*W)
    dW += reg * 2 * W
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
    # the regularization both in loss and in the gradient computations!         #
    #############################################################################
    #                          START OF YOUR CODE                               #
    #############################################################################
    num_train = X.shape[0]
    scores = X.dot(W)
    scores = scores-np.max(scores, axis=1, keepdims=True)
    sum_exp_scores = np.exp(scores).sum(axis=1, keepdims=True)
    softmax_matrix = np.exp(scores)/sum_exp_scores
    loss = np.sum(-np.log(softmax_matrix[np.arange(num_train), y]) )
    softmax_matrix[np.arange(num_train),y] -= 1
    dW = X.T.dot(softmax_matrix)
    loss /= num_train
    dW /= num_train
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W 
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################


    return loss, dW

