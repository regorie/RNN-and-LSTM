import numpy as np
import layers

class RNN_manyToOne:
    def __init__(self, nin, nout, hidden_size, scale, stateful=False, seed=None):
        self.stateful = stateful
        if seed is not None:
            np.random.seed(seed)

        embed_W = np.eye(nin).astype('f') # does nothing but converts labels to one hot vector
        self.rnn_Wx = scale * (np.random.randn(nin, hidden_size)).astype('f')
        self.rnn_Wh = scale * (np.random.randn(hidden_size, hidden_size)).astype('f')
        self.rnn_bx = np.zeros(hidden_size).astype('f')
        self.rnn_Wy = scale * (np.random.randn(hidden_size, nout)).astype('f')
        self.rnn_by = np.zeros(nout).astype('f')

        self.params = [self.rnn_Wx, self.rnn_Wh, self.rnn_Wy,
                       self.rnn_bx, self.rnn_by]

        np.random.seed()

        self.embed_W = embed_W
        self.rnn_layer = [] # this list will be filled with RNN units during forward propagation
        self.loss_layer = layers.SoftmaxWithLoss_unit()

        self.final_h = None

    def forward(self, input_xs, targets):
        """
        input_xs shape : (N, T)
        targets shape : (N, )
        """
        N, T = input_xs.shape
        embeded = self.embed_W[input_xs] # (N, T, D)

        if not self.stateful or self.final_h is None:
            self.final_h = np.zeros(shape=(N, self.rnn_Wh.shape[0])).astype('f')
        
        for t in range(T):
            new_layer = layers.RNN_unit(self.rnn_Wh, self.rnn_Wx, self.rnn_Wy, self.rnn_bx, self.rnn_by)
            self.final_h, output_y = new_layer.forward(embeded[:, t], self.final_h)
            self.rnn_layer.append(new_layer)

        loss = self.loss_layer.forward(output_y, targets)
        return loss

    def predict(self, input_xs):
        """
        input_xs shape : (N, T)
        """
        N, T = input_xs.shape
        embeded = self.embed_W[input_xs] # (N, T, D)

        if not self.stateful or self.final_h is None:
            self.final_h = np.zeros(shape=(N, self.rnn_Wh.shape[0])).astype('f')
        
        for t in range(T):
            new_layer = layers.RNN_unit(self.rnn_Wh, self.rnn_Wx, self.rnn_Wy, self.rnn_bx, self.rnn_by)
            self.final_h, output_y = new_layer.forward(embeded[:, t], self.final_h)

        return output_y

    def backward(self, dout=1):
        
        grads = [np.zeros_like(self.rnn_Wx), 
                 np.zeros_like(self.rnn_Wh),
                 np.zeros_like(self.rnn_Wy),
                 np.zeros_like(self.rnn_bx),
                 np.zeros_like(self.rnn_by)]

        dy = self.loss_layer.backward(dout) # -> (N, D)
        dh = 0.0
        for rnn_unit in reversed(self.rnn_layer):
            dh, dinput_x = rnn_unit.backward(dh, dy)
            dy = np.zeros_like(dy)
            
            grads[0] += rnn_unit.dWx
            grads[1] += rnn_unit.dWh
            grads[2] += rnn_unit.dWy
            grads[3] += rnn_unit.dbx
            grads[4] += rnn_unit.dby

        self.rnn_layer.clear()
        return grads
    
    def reset_state(self):
        self.final_h = None

class LSTM_manyToOne:
    def __init__(self, nin, nout, hidden_size, scale, stateful=False, seed=None):
        self.stateful = stateful
        if seed is not None:
            np.random.seed(seed)

        embed_W = np.eye(nin).astype('f')
        self.lstm_Wx = scale * (np.random.randn(nin, 4*hidden_size)).astype('f')
        self.lstm_Wh = scale * (np.random.randn(hidden_size, 4*hidden_size)).astype('f')
        self.lstm_b = np.zeros(4*hidden_size).astype('f')
        self.lstm_Wy = scale * (np.random.randn(hidden_size, nout)).astype('f')
        self.lstm_by = np.zeros(nout).astype('f')

        self.params = [self.lstm_Wx, self.lstm_Wh, self.lstm_Wy,
                       self.lstm_b, self.lstm_by]
        
        np.random.seed()

        self.embed_W = embed_W
        self.lstm_layer = []
        self.loss_layer = layers.SoftmaxWithLoss_unit()
        
        self.final_h = None
        self.final_c = None

    def forward(self, input_xs, targets):
        """
        input_xs shape : (N, T)
        targets shape : (N, )
        """

        N, T = input_xs.shape
        embedded = self.embed_W[input_xs]

        if not self.stateful or self.final_h is None:
            self.final_h = np.zeros(shape=(N, self.lstm_Wh.shape[0])).astype('f')
            self.final_c = np.zeros(shape=(N, self.lstm_Wh.shape[0])).astype('f')
        
        for t in range(T):
            new_layer = layers.LSTM_unit(self.lstm_Wh, self.lstm_Wx, self.lstm_b, self.lstm_Wy, self.lstm_by)
            self.final_h, self.final_c, output_y = new_layer.forward(embedded[:, t], self.final_h, self.final_c)
            self.lstm_layer.append(new_layer)

        loss = self.loss_layer.forward(output_y, targets)
        return loss
    
    def predict(self, input_xs):
        N, T = input_xs.shape
        embedded = self.embed_W[input_xs]

        if not self.stateful or self.final_h is None:
            self.final_h = np.zeros(shape=(N, self.lstm_Wh.shape[0])).astype('f')
            self.final_c = np.zeros(shape=(N, self.lstm_Wh.shape[0])).astype('f')

        for t in range(T):
            new_layer = layers.LSTM_unit(self.lstm_Wh, self.lstm_Wx, self.lstm_b, self.lstm_Wy, self.lstm_by)
            self.final_h, self.final_c, output_y = new_layer.forward(embedded[:, t], self.final_h, self.final_c)

        return output_y
    
    def backward(self, dout=1):

        grads = [np.zeros_like(self.lstm_Wx),
                 np.zeros_like(self.lstm_Wh),
                 np.zeros_like(self.lstm_Wy),
                 np.zeros_like(self.lstm_b),
                 np.zeros_like(self.lstm_by)]

        dy = self.loss_layer.backward(dout)
        dh = 0.0
        dc = 0.0
        for lstm_unit in reversed(self.lstm_layer):
            dh, dc, dinput_x = lstm_unit.backward(dh, dc, dy)
            dy = np.zeros_like(dy)

            grads[0] += lstm_unit.dWx
            grads[1] += lstm_unit.dWh
            grads[2] += lstm_unit.dWy
            grads[3] += lstm_unit.db
            grads[4] += lstm_unit.dby

        self.lstm_layer.clear()
        return grads
    
    def reset_state(self):
        self.final_h = None
        self.final_c = None

class LSTM97_manyToOne:
    def __init__():
        pass
    
    def forward():
        pass

    def backward():
        pass