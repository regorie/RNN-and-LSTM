import numpy as np
import functions as f
from functions import sigmoid

class RNN_unit:
    def __init__(self, Wh, Wx, Wy, bx, by):
        # set parameters

        self.Wh = Wh
        self.Wx = Wx
        self.Wy = Wy
        self.bx = bx
        self.by = by

        self.dWh = None
        self.dWx = None
        self.dWy = None
        self.dbx = None
        self.dby = None

        # cache
        self.h = None # (batch_size, H)
        self.h_prev = None # (batch_size, H)
        self.input_x = None # (batch_size, D)
        self.output_y = None # (batch_size, D)

    def forward(self, x, h_prev):
        # x shape : (N, D)
        self.h_prev = h_prev
        self.input_x = x

        self.h = np.tanh(np.matmul(h_prev, self.Wh) + np.matmul(x, self.Wx) + self.bx)
        self.output_y = np.matmul(self.h, self.Wy) + self.by
        return self.h, self.output_y
    
    def backward(self, dh, dy):
        self.dWy = np.matmul(self.h.T, dy)
        self.dby = np.sum(dy, axis=0)

        dh += np.matmul(dy, self.Wy.T)

        dtanh = (1 - self.h**2) * dh
        self.dWh = np.matmul(self.h_prev.T, dtanh)
        self.dWx = np.matmul(self.input_x.T, dtanh)
        self.dbx = np.sum(dtanh, axis=0)

        dh_prev = np.matmul(dtanh, self.Wh.T)
        dinput_x = np.matmul(dtanh, self.Wx.T)

        return dh_prev, dinput_x
    
class LSTM_unit:
    def __init__(self, 
                 Wh, Wx, b,
                 Wy, by):
        """
        Wh shape : (H, 4H)
        Wx shape : (Din, 4H)
        Wy shape : (H, Dout)
        """
        self.Wh = Wh
        self.Wx = Wx
        self.b = b
        self.Wy = Wy
        self.by = by

        self.dWh = None
        self.dWx = None
        self.db = None
        self.dWy = None
        self.dby = None

        # cache
        self.c = None
        self.h = None
        self.h_prev = None
        self.c_prev = None
        self.input_x = None
        self.output_y = None

        self.I, self.G, self.F, self.O = None, None, None, None

    def forward(self, input_x, h_prev, c_prev):
        
        N, H = h_prev.shape
        self.h_prev = h_prev
        self.c_prev = c_prev
        self.input_x = input_x

        sum_t = np.matmul(input_x, self.Wx) + np.matmul(h_prev, self.Wh) + self.b
        self.I = sigmoid.f(sum_t[:,    :H]) # 0 - H-1
        self.F = sigmoid.f(sum_t[:,   H:2*H]) # H - 2H-1
        self.O = sigmoid.f(sum_t[:, 2*H:3*H]) # 2H - 3H-1
        self.G = np.tanh(sum_t[:, 3*H:])

        self.c = self.F * c_prev + self.G * self.I
        self.h = self.O * np.tanh(self.c)

        self.output_y = np.matmul(self.h, self.Wy) + self.by

        return self.h, self.c, self.output_y
    
    def backward(self, dh, dc, dy):
        self.dWy = np.matmul(self.h.T, dy)
        self.dby = np.sum(dy, axis=0)

        dh += np.matmul(dy, self.Wy.T)
        dc += dh * self.O * (1 - np.tanh(self.c) ** 2)

        dI = dc * self.G * self.I * (1 - self.I)
        dF = dc * self.c_prev * self.F * (1 - self.F)
        dO = dh * np.tanh(self.c) * self.O * (1 - self.O)
        dG = dc * self.I * (1 - self.G**2)

        dsum_t = np.hstack((dI, dF, dO, dG))

        self.dWh = np.matmul(self.h_prev.T, dsum_t)
        self.dWx = np.matmul(self.input_x.T, dsum_t)
        self.db = np.sum(dsum_t, axis=0)

        dh_prev = np.matmul(dsum_t, self.Wh.T)
        dinput_x = np.matmul(dsum_t, self.Wx.T)
        dc_prev = dc * self.F

        return dh_prev, dc_prev, dinput_x

class LSTM97_unit:
    """
    From 1997 Long Short Term memory
    The first version of LSTM does not have a forget gate, it has only input and output gates
    weights : Wi, Wc, Wo
    """
    def __init__(self, num_of_cell_per_block,
                 Wi, bi,
                 Wo, bo,
                 Wg, bg,
                 Wy, by):
        """
        Wi shape : (16, 2) *(all non-output units, number of all input gates)
        bi shape : (2) *(number of all input gates)
        Wo shape : (16, 2) *(all non-output units, number of all output gates)
        bo shape : (2) *(number of all output gates)
        Wy shape : (4, 4) *(number of hidden states, all output units)
        by shape : (4) *(all output units)
        Wg shape : (16, 4) *(all non-output units, number of cells)
        bg shape : (4) *(number of cells)
        """

        self.num_of_cell_per_block = num_of_cell_per_block

        self.Wi = Wi
        self.Wo = Wo
        self.Wg = Wg
        self.bi = bi
        self.bo = bo
        self.bg = bg

        self.Wy = Wy
        self.by = by

        self.net_in = None
        self.net_out = None
        self.net_cell = None
        self.net_k = None

        self.ds_in = None
        self.ds_cell = None
        self.ds_in_b = None
        self.ds_cell_b = None

        self.dWi = None
        self.dWo = None
        self.dWg = None
        self.dbi = None
        self.dbo = None
        self.dbg = None

        self.dWy = None
        self.dby = None

        # cache
        self.c = None
        self.h = None
        self.state_input = None

        self.I, self.G, self.O = None, None, None

        return
    
    def forward(self, input_x, state_prev, c_prev, h_prev):
        
        c_prev = c_prev.reshape(1, -1)
        h_prev = h_prev.reshape(1, -1)
        state_prev = state_prev.reshape(1, -1)
        input_x = input_x.reshape(1, -1)
        self.state_input = np.hstack((h_prev, state_prev, input_x)) # (1, 16)
        
        # net input to hidden layer
        self.net_in = np.matmul(self.state_input, np.repeat(self.Wi, self.num_of_cell_per_block, axis=1)) + np.repeat(self.bi,self.num_of_cell_per_block) # (1, 4)
        self.net_out = np.matmul(self.state_input, self.Wo) + self.bo # (1, 2)
        self.net_cell = np.matmul(self.state_input, self.Wg) + self.bg # (1, 4)

        # activations in hidden layer
        self.I = sigmoid.f(self.net_in) # (1, 4)
        self.O = sigmoid.f(self.net_out) # (1, 2)

        self.c = c_prev + self.I * sigmoid.g(self.net_cell) # (1, 4)
        self.h = np.repeat(self.O, self.num_of_cell_per_block) * sigmoid.h(self.c) # (1, 4)

        # net input and activations of output units
        self.net_k = np.matmul(self.h, self.Wy) + self.by # (4,) * (4, 4) -> (4,)
        output_y = sigmoid.f(self.net_k)

        # derivatives for input, forget gates and cells
        ## input gate
        self.ds_in = self.ds_in + \
            np.matmul(self.state_input.T, (sigmoid.g(self.net_cell) * sigmoid.df(self.net_in)))
        self.ds_in_b = self.ds_in_b + \
            (sigmoid.g(self.net_cell) * sigmoid.df(self.net_in))

        ## cells
        self.ds_cell = self.ds_cell + \
            np.matmul(self.state_input.T, self.I * sigmoid.dg(self.net_cell))
        self.ds_cell_b = self.ds_cell_b + \
            self.I * sigmoid.dg(self.net_cell)

        state = np.hstack((np.sum(self.I.reshape(-1, self.num_of_cell_per_block), axis=1).reshape(1,-1), self.O))
        return state, self.c, self.h, output_y

    def backward(self, ek):
        ## error and deltas
        # ek = injected error
        dfy = sigmoid.df(self.net_k) * ek # output unit delta k

        dh = np.matmul(dfy, self.Wy.T)
        dO = np.sum((dh * sigmoid.h(self.c)).reshape(-1,self.num_of_cell_per_block), axis=1)
        dnet_out = dO * sigmoid.df(self.net_out) # output gate delta out

        dc = dh * np.repeat(self.O, self.num_of_cell_per_block) * sigmoid.dh(self.c) # input gate, forget gate es (1, 4)
        
        ## weight updates
        # output units and output gates
        self.dWy = np.matmul(self.h.T, dfy)
        self.dby = np.sum(dfy, axis=0)
        self.dWo = np.dot(self.state_input.T, dnet_out)
        self.dbo = np.sum(dnet_out, axis=0)
        
        # input gates
        self.dWi = np.sum((dc * self.ds_in).reshape(-1, self.num_of_cell_per_block), axis=1).reshape(-1, self.num_of_cell_per_block)
        self.dbi = np.sum((dc * self.ds_in_b).reshape(-1, self.num_of_cell_per_block), axis=1).reshape(1, -1).flatten()

        # cells
        self.dWg = dc * self.ds_cell
        self.dbg = (dc * np.sum(self.ds_cell_b, axis=0)).flatten()

        return

class LSTMforget_unit:
    """
    From 2000 Learning to forget: Continual Prediction with LSTM
    Forget gate was introduced to the cell.
    weights : Wi, Wc, Wo, Wf
    """
    def __init__(self, num_of_cell_per_block,
                 Wi, bi,
                 Wo, bo,
                 Wg, bg,
                 Wf, bf,
                 Wy, by):
        """
        Wi shape : (16, 2) *(all non-output units, number of all input gates)
        bi shape : (2) *(number of all input gates)
        Wo shape : (16, 2) *(all non-output units, number of all output gates)
        bo shape : (2) *(number of all output gates)
        Wf shape : (16, 2) *(all non-output units, number of all forget gates)
        bf shape : (2) *(number of all forget gates)
        Wy shape : (4, 4) *(number of hidden states, all output units)
        by shape : (4) *(all output units)
        Wg shape : (16, 4) *(all non-output units, number of cells)
        bg shape : (4) *(number of cells)
        """
        self.num_of_cell_per_block = num_of_cell_per_block

        self.Wi = Wi
        self.Wo = Wo
        self.Wg = Wg
        self.Wf = Wf
        self.bi = bi
        self.bo = bo
        self.bg = bg
        self.bf = bf

        self.Wy = Wy
        self.by = by

        self.net_in = None
        self.net_out = None
        self.net_forget = None
        self.net_cell = None
        self.net_k = None

        self.ds_in = None
        self.ds_cell = None
        self.ds_forget = None
        self.ds_in_b = None
        self.ds_cell_b = None
        self.ds_forget_b = None

        self.dWi = None
        self.dWo = None
        self.dWg = None
        self.dWf = None
        self.dbi = None
        self.dbo = None
        self.dbg = None
        self.dbf = None

        self.dWy = None
        self.dby = None

        # cache
        self.c = None
        self.h = None
        self.state_input = None

        self.I, self.G, self.O, self.F = None, None, None, None

        return

    def forward(self, input_x, state_prev, c_prev, h_prev):
        
        c_prev = c_prev.reshape(1, -1)
        h_prev = h_prev.reshape(1, -1)
        state_prev = state_prev.reshape(1, -1)
        input_x = input_x.reshape(1, -1)
        self.state_input = np.hstack((h_prev, state_prev, input_x)) # (1, 16)
        
        # net input to hidden layer
        self.net_in = np.matmul(self.state_input, np.repeat(self.Wi, self.num_of_cell_per_block, axis=1)) + np.repeat(self.bi,self.num_of_cell_per_block) # (1, 4)
        self.net_forget = np.matmul(self.state_input, np.repeat(self.Wf, self.num_of_cell_per_block, axis=1)) + np.repeat(self.bf,self.num_of_cell_per_block) # (1, 4)
        self.net_out = np.matmul(self.state_input, self.Wo) + self.bo # (1, 2)
        self.net_cell = np.matmul(self.state_input, self.Wg) + self.bg # (1, 4)

        # activations in hidden layer
        self.I = sigmoid.f(self.net_in) # (1, 4)
        self.F = sigmoid.f(self.net_forget) # (1, 4)
        self.O = sigmoid.f(self.net_out) # (1, 2)

        self.c = self.F * c_prev + self.I * sigmoid.g(self.net_cell) # (1, 4)
        self.h = np.repeat(self.O, self.num_of_cell_per_block) * sigmoid.h(self.c) # (1, 4)

        # net input and activations of output units
        self.net_k = np.matmul(self.h, self.Wy) + self.by # (4,) * (4, 4) -> (4,)
        output_y = sigmoid.f(self.net_k)

        # derivatives for input, forget gates and cells
        ## input gate
        self.ds_in = self.ds_in * self.F + \
            np.matmul(self.state_input.T, (sigmoid.g(self.net_cell) * sigmoid.df(self.net_in)))
        self.ds_in_b = self.ds_in_b * self.F + \
            (sigmoid.g(self.net_cell) * sigmoid.df(self.net_in))

        ## forget gate
        self.ds_forget = self.ds_forget * self.F + \
            np.matmul(self.state_input.T, (sigmoid.h(self.c) * sigmoid.df(self.net_forget)))
        self.ds_forget_b = self.ds_forget_b * self.F + \
            (sigmoid.h(self.c) * sigmoid.df(self.net_forget))

        ## cells
        self.ds_cell = self.ds_cell * self.F + \
            np.matmul(self.state_input.T, self.I * sigmoid.dg(self.net_cell))
        self.ds_cell_b = self.ds_cell_b * self.F + \
            self.I * sigmoid.dg(self.net_cell)

        state = np.hstack((np.sum(self.I.reshape(-1, self.num_of_cell_per_block), axis=1).reshape(1,-1), self.O, np.sum(self.F.reshape(-1, self.num_of_cell_per_block), axis=1).reshape(1,-1)))
        return state, self.c, self.h, output_y

    def backward(self, ek):
        ## error and deltas
        # ek = injected error
        dfy = sigmoid.df(self.net_k) * ek # output unit delta k

        dh = np.matmul(dfy, self.Wy.T)
        dO = np.sum((dh * sigmoid.h(self.c)).reshape(-1,self.num_of_cell_per_block), axis=1)
        dnet_out = dO * sigmoid.df(self.net_out) # output gate delta out

        dc = dh * np.repeat(self.O, self.num_of_cell_per_block) * sigmoid.dh(self.c) # input gate, forget gate es (1, 4)
        
        ## weight updates
        # output units and output gates
        self.dWy = np.matmul(self.h.T, dfy)
        self.dby = np.sum(dfy, axis=0)
        self.dWo = np.dot(self.state_input.T, dnet_out)
        self.dbo = np.sum(dnet_out, axis=0)
        
        # input gates
        self.dWi = np.sum((dc * self.ds_in).reshape(-1, self.num_of_cell_per_block), axis=1).reshape(-1, self.num_of_cell_per_block)
        self.dbi = np.sum((dc * self.ds_in_b).reshape(-1, self.num_of_cell_per_block), axis=1).reshape(1, -1).flatten()

        # forget gates
        self.dWf = np.sum((dc * self.ds_forget).reshape(-1, self.num_of_cell_per_block), axis=1).reshape(-1, self.num_of_cell_per_block)
        self.dbf = np.sum((dc * self.ds_forget_b).reshape(-1, self.num_of_cell_per_block), axis=1).reshape(1, -1).flatten()

        # cells
        self.dWg = dc * self.ds_cell
        self.dbg = (dc * np.sum(self.ds_cell_b, axis=0)).flatten()

        return

class SoftmaxWithLoss_unit:
    def __init__(self):
        # cache
        self.y = None
        self.target = None

    def forward(self, input_x, target):
        """
        Considers only the last output
        input_x shape : (batch_size, D)
        target shape : (batch_Size, ) -> converts to (batch_size, D)
        """
        N, D = input_x.shape

        if target.ndim == 1:
            # convert to one-hot
            tmp = np.zeros_like(input_x)
            tmp[np.arange(N), target] = 1.0
            target = tmp

        y = f.softmax(input_x)

        # calculate loss
        loss = -(target*np.log(y)).mean(axis=0).sum()

        # save cache
        self.y = y
        self.target = target

        return loss
    
    def backward(self, dout=1):
        """
        output dx shape : (N, D)
        """
        N, D = self.target.shape

        dx = self.y.copy() * dout
        dx[np.arange(N), np.argmax(self.target, axis=-1)[np.arange(N)]] -= 1.0

        return dx/N

class SEloss_unit:
    def __init__(self):
        self.output_y = None
        self.target = None

    def forward(self, output_y, target):
        """
        input_x shape : (Dout)
        """
        self.output_y = output_y
        self.target = target

        loss = (target - output_y)**2
        loss = np.sum(loss) / 2
        return loss

    def backward(self, dout=1):
        """
        output dx shape : (Dout)
        """
        dx = -(self.target - self.output_y)
        return dx