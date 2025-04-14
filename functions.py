import numpy as np

def softmax(x):
    """
    input x shape : (N, D)
    output shape : (N, D)
    """
    N, D = x.shape
    x -= np.max(x, axis=1,keepdims=True)

    exp_x = np.exp(x) 
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True) # (N, 1)
    return exp_x / sum_exp_x


class sigmoid: # range [-2, 2]
    @staticmethod
    def f(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def h(x):  # range [-1, 1]
        return 2 / (1 + np.exp(-x)) - 1
    
    @staticmethod
    def g(x): # range[-2, 2]
        return 4 / (1 + np.exp(-x)) - 2
    
    @staticmethod
    def df(x):
        f = 1 / (1 + np.exp(-x))
        return f * (1 - f)
    
    @staticmethod
    def dh(x):
        f = 1 / (1 + np.exp(-x))
        return 2 * f * (1 - f)
    
    @staticmethod
    def dg(x):
        f = 1 / (1 + np.exp(-x))
        return 4 * f * (1 - f)

def cross_entropy_error(y, t):
    """
    y shape : (N, C)
    t shape : (N, C) : target given as one-hot vector
              (N,) : target given as labels
    """
    batch_size = y.shape[0]
    if y.ndim == t.ndim:
        t = np.argmax(t, axis=1)

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size