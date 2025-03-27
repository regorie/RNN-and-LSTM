import numpy as np


def clip_grads(grads, thres=1.0):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = thres / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate