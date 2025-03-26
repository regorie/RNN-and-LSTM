import numpy as np
from layers import *

class RNNLM:
    def __init__(self, vocab_size, hidden_size, seed):

        if seed is not None:
            np.random.seed(seed)

        scale = 0.1
        #embed_W = (np.random.randn(vocab_size, embed_size) / 100).astype('f')
        embed_W = np.eye(vocab_size).astype('f')
        rnn_Wx = scale * (np.random.randn(vocab_size, hidden_size)).astype('f')
        rnn_Wh = scale * (np.random.randn(hidden_size, hidden_size)).astype('f')
        rnn_b = (np.zeros(hidden_size)).astype('f')
        affine_W = scale * (np.random.randn(hidden_size, vocab_size)).astype('f')
        affine_b = (np.zeros(vocab_size)).astype('f')

        #self.embed_layer = TimeEmbedding(embed_W)
        self.embed_W = embed_W
        self.rnn_layer = TimeRNN(rnn_Wh, rnn_Wx, rnn_b)
        self.affine_layer = TimeAffine(affine_W, affine_b)
        self.loss_layer = TimeSoftmaxWithLoss_manytoone()

        self.params = {}
        #self.params['Embed'] = self.embed_layer.W
        self.params['Rnn_Wh'] = self.rnn_layer.Wh
        self.params['Rnn_Wx'] = self.rnn_layer.Wx
        self.params['Rnn_b'] = self.rnn_layer.b
        self.params['Affine_W'] = self.affine_layer.W
        self.params['Affine_b'] = self.affine_layer.b

        # cache
        self.loss = None
        self.h_prev = None

    def forward(self, x, t):
        """
        x shape : (N, T, D)
                  (N, T,)
        t shape : (N, T, D)
                  (N, T,)
                  (N, )
        """

        out = self.embed_W[x]
        out = self.rnn_layer.forward(out, self.h_prev)
        self.h_prev = out[:, -1, :].copy()
        out = self.affine_layer.forward(out)

        self.loss = self.loss_layer.forward(out, t)
        weight_decay = 0.0
        

        return self.loss

    def generate(self, x):
        out = self.embed_W[x]
        out = self.rnn_layer.forward(out, None)
        out = self.affine_layer.forward(out)

        return out

    def backward(self, dout=1):

        dout = self.loss_layer.backward(dout)
        dout = self.affine_layer.backward(dout)
        dout = self.rnn_layer.backward(dout)

        grads = {}
        grads['Rnn_Wh'] = self.rnn_layer.dWh
        grads['Rnn_Wx'] = self.rnn_layer.dWx
        grads['Rnn_b'] = self.rnn_layer.db
        grads['Affine_W'] = self.affine_layer.dW
        grads['Affine_b'] = self.affine_layer.db

        return grads
    
    def reset_state(self):
        self.h_prev = None


class LSTMLM:
    def __init__(self, vocab_size, hidden_size, seed):
        # initialize weights

        if seed is not None:
            np.random.seed(seed)

        scale = 0.1
        #embed_w = (np.random.randn(vocab_size, embed_size) / 100).astype('f')
        embed_W = np.eye(vocab_size).astype('f')
        lstm_Wx = scale * (np.random.randn(vocab_size, 4 * hidden_size)).astype('f')
        lstm_Wh = scale * (np.random.randn(hidden_size, 4 * hidden_size)).astype('f')
        lstm_b = np.zeros(4* hidden_size).astype('f')
        affine_W = scale * (np.random.randn(hidden_size, vocab_size)).astype('f')
        affine_b = np.zeros(vocab_size).astype('f')

        # build layers
        #self.embed_layer = TimeEmbedding(embed_w)
        self.embed_W = embed_W
        self.lstm_layer = TimeLSTM(lstm_Wh, lstm_Wx, lstm_b)
        self.affine_layer = TimeAffine(affine_W, affine_b)
        self.loss_layer = TimeSoftmaxWithLoss_manytoone()

        self.params = {}
        #self.params['Embed'] = self.embed_layer.W
        self.params['LSTM_Wh'] = self.lstm_layer.Wh
        self.params['LSTM_Wx'] = self.lstm_layer.Wx
        self.params['LSTM_b'] = self.lstm_layer.b
        self.params['Affine_W'] = self.affine_layer.W
        self.params['Affine_b'] = self.affine_layer.b

        # cache
        self.loss = None
        self.h_prev = None
        self.c_prev = None


    def forward(self, x, t):
        """
        x shape : (N, T, D)
                  (N, T,)
        t shape : (N, T, D)
                  (N, T,)
        """
        #out = self.embed_layer.forward(x)
        out = self.embed_W[x]
        out, cout = self.lstm_layer.forward(out, self.h_prev, self.c_prev)
        self.h_prev = out[:, -1, :].copy()
        self.c_prev = cout
        out = self.affine_layer.forward(out)

        self.loss = self.loss_layer.forward(out, t)
        return self.loss

    def generate(self, x):
        #out = self.embed_layer.forward(x)
        out = self.embed_W[x]
        out, cout = self.lstm_layer.forward(out, None, None)
        out = self.affine_layer.forward(out)

        return out

    def backward(self, dout=1):

        dout = self.loss_layer.backward(dout)
        dout = self.affine_layer.backward(dout)
        dout = self.lstm_layer.backward(dout)
        #self.embed_layer.backward(dout)

        grads = {}
        #grads['Embed'] = self.embed_layer.dW
        grads['LSTM_Wh'] = self.lstm_layer.dWh
        grads['LSTM_Wx'] = self.lstm_layer.dWx
        grads['LSTM_b'] = self.lstm_layer.db
        grads['Affine_W'] = self.affine_layer.dW
        grads['Affine_b'] = self.affine_layer.db

        return grads

    def reset_state(self):
        self.h_prev = None
        self.c_prev = None