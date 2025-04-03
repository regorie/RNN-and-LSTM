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

class LSTM_unit_legacy:
    """
    From 1997 Long Short Term memory
    The first version of LSTM does not have a forget gate, it has only input and output gates
    weights : Wi, Wc, Wo
    """
    def __init__(self, num_of_cell_per_block, num_of_memblock,
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
        
        self.num_of_total_cells = num_of_cell_per_block * num_of_memblock
        self.num_of_cell_per_block = num_of_cell_per_block

        self.Wi = Wi
        self.Wo = Wo
        self.Wg = Wg
        self.bi = bi
        self.bo = bo
        self.bg = bg

        self.Wy = Wy
        self.by = by

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
        self.c_prev = None
        self.h = None
        self.state = None
        self.state_prev = None
        self.input_x = None

        self.output_y = None

        self.I, self.G, self.O = None, None, None, None

        return

    def forward(self, input_x, state_prev, c_prev):
        """
        input_x shape : (Din)
        state_prev shape : (8) *(non-in/output units)
        c_prev shape : (num_of_total_cell)
        """
        self.state_prev = state_prev
        self.c_prev = c_prev
        self.input_x = input_x

        state_input = np.hstack(state_prev, input_x)

        self.I = f.legacy_f(np.matmul(state_input, self.Wi.T) + self.bi) 
        self.O = f.legacy_f(np.matmul(state_input, self.Wo.T) + self.bo)
        self.G = f.legacy_g(np.matmul(state_input, self.Wg.T) + self.bg)

        IG = self.I * self.G
        self.c = self.c_prev + np.repeat(IG, 2)
        self.h = self.O * f.legacy_h(self.c)
       
        self.h = self.h.flatten()
        self.output_y = np.matmul(self.h, self.Wy) + self.by
        
        self.state = np.hstack(self.h, self.I, self.O)
        return self.state, self.c, self.output_y

    def backward(self, dh, dc, dy):
        """
        In 1997 LSTM, the backward pass only occurs at the last timestep(No error flow through time).
        dh, dc shape : (num_of_memblocks, num_of_cells_per_block, batch_size, H)
        dy shape : (batch_size, Dout)
        """
        dh_prev = np.empty_like(dh)
        dc_prev = np.empty_like(dc)
        dinput_x = np.empty_like(self.input_x)

        self.dby = np.sum(dy, axis=0)
        self.dWy = np.matmul(self.concat_h.T, dy)

        dh += np.matmul(dy, self.Wy.T).reshape(dh.shape)

        for i, memblock in enumerate(self.memblocks):
            dh_prev[i], dc_prev[i], tmp = memblock.backward(dh[i], dc[i])
            dinput_x += tmp

        return dh_prev, dc_prev, dinput_x

class LSTM_unit_forgetgate:
    """
    From 2000 Learning to forget
    Forget gate was introduced to the cell.
    weights : Wi, Wc, Wo, Wf
    """
    def __init__(self, num_of_cell, num_of_memblock,
                 Wh, Wx, b,
                 Wy, by):
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

    def backward(self):
        pass

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
