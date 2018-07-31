import numpy as np


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
    n = X.shape[0]
    num_features = X.shape[1]
    num_classes = W.shape[1]

    ###########################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.   #
    # Store the loss in loss and the gradient in dW. If you are not careful   #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                                         #
    ###########################################################################
    scores = X @ W
    # scaling (on terms of exponential) for numerical stability
    # (see https://cs231n.github.io/linear-classify/#softmax)
    scores -= np.max(scores, axis=1).reshape((-1, 1))
    correct_class_scores = scores[range(n), y].reshape((-1, 1))

    # compute the loss
    for i in range(n):
        loss_i = (-1) * correct_class_scores[i] \
            + np.log(np.sum(np.exp(scores[i])))
        loss += loss_i / n
        for j in range(num_features):
            for k in range(num_classes):
                dLi_dWjk = 1 / np.sum(np.exp(scores[i])) \
                    * np.exp(scores[i, k]) \
                    * X[i, j]
                if k == y[i]:
                    dLi_dWjk -= X[i, j]
                dW[j, k] += dLi_dWjk / n

    loss += reg * np.sum(W ** 2)
    dW += 2 * W
    ###########################################################################
    #                          END OF YOUR CODE                               #
    ###########################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    n = X.shape[0]
    num_features = X.shape[1]
    num_classes = W.shape[1]

    scores = X @ W
    # scaling (on terms of exponential) for numerical stability
    # (see https://cs231n.github.io/linear-classify/#softmax)
    scores -= np.max(scores, axis=1).reshape((-1, 1))
    correct_class_scores = scores[range(n), y].reshape((-1, 1))

    loss_per_obs = (-1) * correct_class_scores \
        + np.log(np.sum(np.exp(scores), axis=1)).reshape((-1, 1))
    loss = np.sum(loss_per_obs) / n + reg * np.sum(W ** 2)

    softmax_scores_rowsums = np.sum(np.exp(scores), axis=1).reshape((-1, 1))
    softmax_scores = np.exp(scores) / softmax_scores_rowsums

    # the (i, j, k)th entry of M is dL_i / dW_{j, k}
    M = softmax_scores.reshape((n, 1, num_classes)) \
        * X.reshape((n, num_features, 1))
    M[range(n), :, y] = M[range(n), :, y] - X
    dW = np.sum(M, axis=0) / n
    dW += 2 * W

    return loss, dW
