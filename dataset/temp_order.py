import random
import numpy as np
import argparse

def generate_batch(sequence_length=25, batch_size=3):

    l = sequence_length
    p0 = np.random.randint(low=0, high=l*0.1, size=(batch_size, )) + int(l*0.1)
    p1 = np.random.randint(low=0, high=l*0.1, size=(batch_size,)) + int(l*0.1)
    v0 = np.random.randint(low=0, high=2, size=(batch_size,))
    v1 = np.random.randint(low=0, high=2, size=(batch_size,))

    targets = v0 + v1*2

    data = np.random.randint(low=0, high=4, size=(l, batch_size)) + 2
    data[p0, np.arange(batch_size)] = v0
    data[p1, np.arange(batch_size)] = v1

    return data.T, targets # (batch_size, sequence_length), (batch_size)

"""
    # if length = 25 batchsize = 3
    l = length
    p0 = self.rng.randint(int(l*.1), size=(batchsize,)) + int(l*.1)
    # [2.5, 5.0) sequence length (batch_size, ) -> first position
    v0 = self.rng.randint(2, size=(batchsize,))
    # [0, 1] sequence length (batch_size, ) -> second position
    p1 = self.rng.randint(int(l*.1), size=(batchsize,)) + int(l*.5)
    # [12.5, 15) sequence length (batch_size, )
    v1 = self.rng.randint(2, size=(batchsize,))
    # [0, 1] sequence length (batch_size)
    targ_vals = v0 + v1*2
    # (batch_size, ) range [0, 3]
    vals  = self.rng.randint(4, size=(l, batchsize))+2
    # sequence (length, batch_size) range [2, 6)
    vals[p0, numpy.arange(batchsize)] = v0 # values for first pos
    vals[p1, numpy.arange(batchsize)] = v1 # values for second pos

    # turning into one hot vectors
    data = numpy.zeros((l, batchsize, 6), dtype=self.floatX)
    # sequence (length, batch_size, 6) all zeros
    targ = numpy.zeros((batchsize, 4), dtype=self.floatX)
    # sequence (batch_size, 4) all zeros
    data.reshape((l*batchsize, 6))[numpy.arange(l*batchsize),
                                vals.flatten()] = 1.
    targ[numpy.arange(batchsize), targ_vals] = 1.
    return data, targ
"""