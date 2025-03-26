import random
import string
import numpy as np
import pickle
import argparse

# Token vocabulary
vocab = list(string.printable)
vocab_size = len(vocab)
token_to_id = {token: idx for idx, token in enumerate(vocab)}
id_to_token = {idx: token for token, idx in token_to_id.items()}

def generate_sequence(sample_num, seq_len=10, match_pos=0, is_random=False):
    

    if is_random:
        input_ids = np.empty(shape=(sample_num, sequence_length-1), dtype=np.int32)
        target_ids = np.empty(shape=(sample_num, sequence_length-1), dtype=np.int32)

        test_input_ids = np.empty(shape=(1000, sequence_length-1), dtype=np.int32)
        test_target_ids = np.empty(shape=(1000, sequence_length-1), dtype=np.int32)
        
        # generate train data
        for i in range(sample_num):
            sequence = [random.choice(vocab) for _ in range(seq_len)]
            sequence = [token_to_id[token] for token in sequence]

            sequence[-1] = sequence[0]
            input_ids[i,:] = sequence[:-1]
            target_ids[i,:] = sequence[1:]
        
        # generate test data
        for i in range(1000):
            sequence = [random.choice(vocab) for _ in range(seq_len)]
            sequence = [token_to_id[token] for token in sequence]

            sequence[-1] = sequence[0]
            test_input_ids[i,:] = sequence[:-1]
            test_target_ids[i,:] = sequence[1:]
            
    else:
        if sample_num > vocab_size : sample_num = vocab_size

        input_ids = np.empty(shape=(sample_num, sequence_length-1), dtype=np.int32)
        target_ids = np.empty(shape=(sample_num, sequence_length-1), dtype=np.int32)

        test_input_ids = np.empty(shape=(vocab_size, sequence_length-1), dtype=np.int32)
        test_target_ids = np.empty(shape=(vocab_size, sequence_length-1), dtype=np.int32)

        subsequence = [random.choice(vocab) for _ in range(seq_len-2)]
        subsequence = [token_to_id[token] for token in subsequence]

        # generate train data
        for i in range(sample_num):
            sequence = [token_to_id[vocab[i]]] + subsequence + [token_to_id[vocab[i]]]
            input_ids[i,:] = sequence[:-1]
            target_ids[i,:] = sequence[1:]

        # generate test data
        for i in range(vocab_size):
            sequence = [token_to_id[vocab[i]]] + subsequence + [token_to_id[vocab[i]]]
            test_input_ids[i,:] = sequence[:-1]
            test_target_ids[i,:] = sequence[1:]


    return input_ids, target_ids, test_input_ids, test_target_ids


def load_data(len=10, train=True, is_random=True):
    if train: 
        with open('./dataset/synthesize{}_random_subsequnce_{}.train.pkl'.format(len, str(is_random)), 'rb') as f:
            data = pickle.load(f)
    else:
        with open('./dataset/synthesize{}_random_subsequnce_{}.test.pkl'.format(len, str(is_random)), 'rb') as f:
            data = pickle.load(f)
    
    return data['inputs'], data['targets']


# Example
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--time_lag", '-lag', type=int, required=False, default=4)
    parser.add_argument("--is_random", '-rand', type=int, default=0)
    parser.add_argument("--sample_num", '-sn', type=int, default=10000)
    args = parser.parse_args()

    sequence_length = args.time_lag + 2
    sample_num = args.sample_num
    if args.is_random == 1: is_random = True
    else : is_random = False

    input_ids, target_ids, test_input_ids, test_target_ids = generate_sequence(sample_num, seq_len=sequence_length, match_pos=0, is_random=is_random)

    with open('./synthesize{}_random_subsequnce_{}.train.pkl'.format(sequence_length, str(is_random)), 'wb') as f:
        pickle.dump({'inputs' : input_ids, 
                     'targets': target_ids,
                     'token_to_id': token_to_id,
                     'id_to_token': id_to_token}, f)
        
    with open('./synthesize{}_random_subsequnce_{}.test.pkl'.format(sequence_length, str(is_random)), 'wb') as f:
        pickle.dump({'inputs' : test_input_ids, 
                     'targets': test_target_ids,
                     'token_to_id': token_to_id,
                     'id_to_token': id_to_token}, f)
