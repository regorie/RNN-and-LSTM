import numpy as np

class SGD:
    def __init__(self, optimizer_params={'lr': 0.01}):
        self.lr = optimizer_params['lr']

    def update(self, grads, params):
        for key in params.keys():
            params[key] -= self.lr * grads[key]