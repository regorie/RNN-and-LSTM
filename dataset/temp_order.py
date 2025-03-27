import random
import numpy as np
import argparse

def generate_batch(sequence_length=25, batch_size=3):

    l = sequence_length
    p0 = np.random.randint(low=0, high=l*0.1, size=(batch_size, )) + int(l*0.1)
    p1 = np.random.randint(low=0, high=l*0.1, size=(batch_size,)) + int(l*0.5)
    v0 = np.random.randint(low=0, high=2, size=(batch_size,))
    v1 = np.random.randint(low=0, high=2, size=(batch_size,))

    targets = v0 + v1*2

    data = np.random.randint(low=0, high=4, size=(l, batch_size)) + 2
    data[p0, np.arange(batch_size)] = v0
    data[p1, np.arange(batch_size)] = v1

    return data.T, targets # (batch_size, sequence_length), (batch_size)
