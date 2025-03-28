import numpy as np
import functions as f

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
        Wx shape : (D, 4H)
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
        self.I = f.sigmoid(sum_t[:,    :H]) # 0 - H-1
        self.F = f.sigmoid(sum_t[:,   H:2*H]) # H - 2H-1
        self.O = f.sigmoid(sum_t[:, 2*H:3*H]) # 2H - 3H-1
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
