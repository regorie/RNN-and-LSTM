import random
import numpy as np
import argparse

def generate_batch(seq_length_range=[100, 110], 
                   pos0_range=[10, 20],
                   pos1_range=[50, 60],
                   label_num=8,
                   batch_size=1):

    # set positions
    p0 = np.random.randint(low=pos0_range[0], high=pos0_range[1]+1, size=(batch_size))
    p1 = np.random.randint(low=pos1_range[0], high=pos1_range[1]+1, size=(batch_size))

    # set special values
    v0 = np.random.randint(low=0, high=2, size=(batch_size,))
    v1 = np.random.randint(low=0, high=2, size=(batch_size,))

    # set targets
    # v0, v1 pairs and corresponding target label :
    #   (0, 0) -> 0, (0, 1) -> 2, (1, 0) -> 1, (1, 1) -> 3
    targets = v0 + v1*2

    # generate sequence
    # input labels:
    # 0, 1 - special values
    # 2, 3 - start, stop triggers
    # 4, 5, 6, 7 - normal values
    seq_len = np.random.randint(low=seq_length_range[0], high=seq_length_range[1]+1)
    data = np.random.randint(low=0, high=label_num-4, size=(batch_size, seq_len)) + 4
    data[np.arange(batch_size), p0] = v0
    data[np.arange(batch_size), p1] = v1
    data[np.arange(batch_size), 0] = 2
    data[np.arange(batch_size), -1] = 3

    return data.T, targets # (batch_size, sequence_length), (batch_size)
