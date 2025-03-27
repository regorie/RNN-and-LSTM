import numpy as np

class SGD:
    def __init__(self, optimizer_params={'lr': 0.01}):
        self.lr = optimizer_params['lr']

    def update(self, grads, params):
        for i, param in enumerate(params):
            param -= self.lr * grads[i]