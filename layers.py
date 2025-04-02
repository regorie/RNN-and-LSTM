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


class _MemBlock:
    def __init__(self, num_cell, Wh, Wx, b, Wgh, Wgx, bg):
        """
        Wh shape : (H, 2H) -> [input gate, output gate]
        Wx shape : (Din, 2H) -> [input gate, output gate]
        Wgh shape : (num_cell, H, H)
        Wgx shape : (num_cell, Din, H)
        """
        self.Wih = Wh[:,:Wh.shape[1]//2]
        self.Wix = Wx[:,:Wx.shape[1]//2]
        self.bi = b[:b.shape[0]//2]

        self.Woh = Wh[:,Wh.shape[1]//2:]
        self.Wox = Wx[:,Wx.shape[1]//2:]
        self.bo = b[b.shape[0]//2:]

        self.Wgh = Wgh # (num_cell, H, H)
        self.Wgx = Wgx
        self.bg = bg

        self.dWih, self.dWix, self.dbi = None, None, None
        self.dWoh, self.dWox, self.dbo = None, None, None
        self.dWgh, self.dWgx, self.dbg = None, None, None

        self.num_cell = num_cell
        self.cells = None

        # cache
        self.I, self.O, self.C, self.G, self.H = [], [], [], [], []

    def forward(self, input_x, h_prev, c_prev):
        """
        input_x shape : (batch_size, Din)
        h_prev shape : (num_of_cells, batch_size, H)
        c_prev shape : (num_of_cells, batch_size, H)
        """
        self.input_x = input_x
        self.h_prev = h_prev
        self.c_prev = c_prev
        batch_size = input_x.shape[0]
        H = h_prev.shape[-1]

        self.I = np.empty((self.num_cell, batch_size, H))
        self.O = np.empty((self.num_cell, batch_size, H))
        self.G = np.empty((self.num_cell, batch_size, H))
        self.C = np.empty((self.num_cell, batch_size, H))
        self.H = np.empty((self.num_cell, batch_size, H))
        for cell_idx in range(self.num_cell):
            # input gate
            self.I[cell_idx] = f.sigmoid(np.matmul(h_prev[:,cell_idx], self.Wih) + np.matmul(input_x, self.Wix) + self.bi)
            # output gate
            self.O[cell_idx] = f.sigmoid(np.matmul(h_prev[:,cell_idx], self.Woh) + np.matmul(input_x, self.Wox) + self.bo)
            # cell update
            self.G[cell_idx] = f.legacy_g(np.matmul(h_prev[:,cell_idx], self.Wgh[cell_idx]) + np.matmul(input_x, self.Wgx[cell_idx]) + self.bg[cell_idx])
            self.C[cell_idx] = c_prev[cell_idx] + self.I[cell_idx] * self.G[cell_idx]
            # hidden state
            self.H[cell_idx] = self.O[cell_idx] * f.legacy_h(self.C[cell_idx])

        return self.H, self.C # (num_of_cell_per_memblock, batch_size, H)

    def backward(self, dh, dc):
        """
        dh, dc shape : (num_of_cell, batch_size, H)
        dh_prev, dc_prev shape : (num_of_cell, batch_size, H)
        dinput_x shape : (batch_size, Din)
        """
        dh_prev = np.empty_like(dc)
        dc_prev = np.empty_like(dh)
        dinput_x = np.zeros_like(self.input_x)
        for cell_idx in range(self.num_cell):
            dc[cell_idx] += dh[cell_idx] * self.O[cell_idx] * (1 - np.tanh(self.C[cell_idx])**2)

            dI = dc[cell_idx] * self.G[cell_idx] * self.I * (1 - self.I[cell_idx])
            dO = dh[cell_idx] * np.tanh(self.C[cell_idx]) * self.O[cell_idx] * (1 - self.O[cell_idx])
            dG = dc[cell_idx] * self.I[cell_idx] * (1 - self.G[cell_idx]**2)

            self.dWih += np.matmul(self.h_prev[:,cell_idx].T, dI)
            self.dWix += np.matmul(self.input_x.T, dI)
            self.dbi += np.sum(dI, axis=0)

            self.dWoh += np.matmul(self.h_prev[:,cell_idx].T, dO)
            self.dWox += np.matmul(self.input_x.T, dO)
            self.dbo += np.sum(dO, axis=0)

            self.dWgh[cell_idx] = np.matmul(self.h_prev[:,cell_idx].T, dG)
            self.dWgx[cell_idx] = np.matmul(self.input_x.T, dG)
            self.dbg[cell_idx] = np.sum(dG, axis=0)

            dh_prev[cell_idx] = np.matmul(dI, self.Wih.T)
            dh_prev[cell_idx] += np.matmul(dO, self.Woh.T)
            dh_prev[cell_idx] += np.matmul(dG, self.Wgh[cell_idx].T)

            dinput_x += np.matmul(dI, self.Wix.T)
            dinput_x += np.matmul(dO, self.Wox.T)
            dinput_x += np.matmul(dG, self.Wgx[cell_idx].T)

            dc_prev[cell_idx] = dc[cell_idx]

        return dh_prev, dc_prev, dinput_x


class LSTM_unit_legacy:
    """
    From 1997 Long Short Term memory
    The first version of LSTM does not have a forget gate, it has only input and output gates
    weights : Wi, Wc, Wo
    """
    def __init__(self, num_of_cell_per_block, num_of_memblock,
                 Wh, Wx, b,
                 Wgh, Wgx, bg,
                 Wy, by):
        """
        Wh shape : (num_of_memblock, H, 2H)
        Wx shape : (num_of_memblock, Din, 2H)
        Wgh shape : (num_of_memblock, num_of_cell_per_block, H, H)
        Wgx shape : (num_of_memblock, num_of_cell_per_block, Din, H)
        Wy shape : (num_of_memblock * num_of_cell_per_block * H, Dout)
        """
        
        self.num_of_total_cells = num_of_cell_per_block * num_of_memblock
        self.num_of_cell_per_block = num_of_cell_per_block

        self.memblocks = []
        for m in range(num_of_memblock):
            new_block = _MemBlock(num_of_cell_per_block, Wh[m], Wx[m], b[m], 
                                  Wgh[m], Wgx[m], bg[m])
            self.append(new_block)

        self.Wy = Wy
        self.by = by

        self.dWy = None
        self.dby = None

        # cache
        self.c = None
        self.h = None
        self.concat_h = None
        self.h_prev = None
        self.c_prev = None
        self.input_x = None
        self.output_y = None

        self.I, self.G, self.O = None, None, None, None

        return

    def forward(self, input_x, h_prev, c_prev):
        """
        input_x shape : (batch_size, Din)
        h_prev shape : (num_of_memblocks, num_of_cells_per_block, batch_size, H)
        c_prev shape : (num_of_memblocks, num_of_cells_per_block, batch_size, H)
        """
        batch_size = input_x.shape[0]
        self.h_prev = h_prev
        self.c_prev = c_prev
        self.input_x = input_x

        self.h = np.empty_like(h_prev)
        self.c = np.empty_like(c_prev)
        for i, memblock in enumerate(self.memblocks):
            self.h[i], self.c[i] = memblock.forward(input_x, h_prev[i], c_prev[i]) # (num_of_cell_per_memblock, batch_size, H)

        # concat hidden states -> shape into (batch_size, num_of_memblocks * num_of_cells_per_block * H)
        self.concat_h = self.h.reshape(batch_size, -1)
        self.output_y = np.matmul(self.concat_h, self.Wy) + self.by

        return self.h, self.c, self.output_y

    def backward(self, dh, dc, dy):
        """
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
