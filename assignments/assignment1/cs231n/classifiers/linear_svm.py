import numpy as np


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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i, :]
                dW[:, y[i]] -= X[i, :]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * 2.0 * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    num_train = X.shape[0]
    num_features = X.shape[1]
    num_classes = W.shape[1]

    # 1. compute the loss

    loss = 0.0
    scores = X @ W
    correct_class_scores = scores[range(num_train), y].reshape((-1, 1))
    margin = np.maximum(0.0, scores - correct_class_scores + 1)
    margin[range(num_train), y] = 0.0
    loss = margin.sum(axis=1).mean() + reg * np.sum(W * W)

    # 2. compute the gradient

    dW = np.zeros(W.shape)  # initialize the gradient as zero
    dmargin_dW = X
    dmargin_dW = dmargin_dW.reshape((1, num_train, num_features))
    dmargin_dW = np.ones((num_classes, 1, 1)) * dmargin_dW
    # (now dmargin_dW is num_classes * num_train * num_features)
    mask = margin.copy()  # margin is num_train * num_classes
    mask[margin > 0.0] = 1.0
    mask[range(num_train), y] = (-1.0) * np.sum(mask, axis=1)

    dmargin_dW *= mask.T.reshape((num_classes, num_train, 1))
    # sum along num_train
    dW = dmargin_dW.sum(axis=1).T
    # (now dW is num_features * num_classes)
    # These last two lines can be done more efficiently as
    # dW = X.T @ mask
    # which would also make some of the previous lines redundant.
    # But that may be too tricky for my tastes right now...

    dW /= num_train
    dW += reg * 2.0 * W

    return loss, dW
