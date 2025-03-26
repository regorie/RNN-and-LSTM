import random
import numpy as np
import argparse

def generate_batch(sequence_length=25, batch_size=3):
    randvals = np.random.randint(low=2,high=100,size=(sequence_length+1, batch_size))
    val = np.random.randint(low=0, high=2, size=(batch_size,))

    randvals[np.zeros(shape=(batch_size,), dtype=np.int32), np.arange(batch_size)] = val
    randvals[np.ones(shape=(batch_size,), dtype=np.int32)*sequence_length, np.arange(batch_size)] = val

    inputs = randvals[:-1]
    targets = np.zeros(shape=(batch_size,))
    targets = randvals[-1]

    return inputs.T, targets # (batch_size, sequence_length), (batch_size)