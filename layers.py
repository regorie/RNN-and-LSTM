import numpy as np
from functions import *

#######################################
## Layers with learnable parameters

class SingleRNN:
    def __init__(self, Wh, Wx, b):
        
        # set parameters
        self.Wh = Wh
        self.Wx = Wx
        self.b = b

        self.dWx = None
        self.dWh = None
        self.db = None

        # cache
        self.h_next = None
        self.x = None
        self.h_prev = None

    def forward(self, x, h_prev):
        tmp = np.dot(h_prev, self.Wh) + np.dot(x, self.Wx) + self.b
        self.h_next = np.tanh(tmp)

        return self.h_next

    def backward(self, dh_next):
        dh = (1 - self.h_next**2) * dh_next

        self.dWx = np.dot(self.x.T, dh)
        dx = np.dot(dh, self.Wx.T)

        self.dWh = np.dot(self.h_prev.T, dh)
        dh_prev = np.dot(dh, self.Wh.T)

        self.db = np.sum(dh, axis=0)

        return dx, dh_prev


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x

        output = np.dot(x, self.W)
        if self.b is None:
            return output
        else:
            return output + self.b

    def backward(self, dout):
        self.dW = np.dot(self.x.T, dout)
        if self.b is not None:
            self.db = np.sum(dout, axis=0)

        dx = np.dot(dout, self.W.T)
        return dx


class Embedding:
    def __init__(self, W):
        """
        W shape : (V, D)
        """
        self.W = W
        self.dW = None
        self.idx = None

    def forward(self, idx):
        """
        idx shape : (N, V) : idx given as one hot vector
                    (N,) : idx given as numbered labels
        output shape : (N, D)
        """
        if idx.ndim == 2:
            idx = np.argmax(idx, axis=1)

        self.idx = idx
        return self.W[idx]

    def backward(self, dout):
        """
        dout shape : (N, D)
        """
        self.dW = np.zeros_like(self.W)

        for i, idx in enumerate(self.idx): 
            self.dW[idx] += dout[i]

        return 
    
#############################################
## Loss layers

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.t = None
        self.y = None

    def forward(self, x, t):
        y = softmax(x)
        self.loss = cross_entropy_error(y, t)
        self.t, self.y = t, y

        return self.loss

    def backward(self, dout=1):
        N = self.y.shape[0]
        
        if self.t.ndim == self.y.ndim:
            return (self.y - self.t) / N
        else:
            dx = self.y.copy()
            dx[np.arange(N), self.t] -= 1.0
            return dx / N

#############################################
## Time layers

class TimeRNN:
    def __init__(self, Wh, Wx, b):
        self.Wh = Wh
        self.Wx = Wx
        self.b = b

        self.dWx = None
        self.dWh = None
        self.db = None

        # cache
        self.h = None
        self.h_prev = None
        self.x = None

    def forward(self, x, h_prev=None):
        """
        x shape : (N, T, D)
        output shape : (N, T, H)
        """
        N, T_, D = x.shape
        H = self.Wh.shape[0]

        self.x = x
        if h_prev is None:
            h_prev = np.zeros((N, H))
        self.h_prev = h_prev
        self.h = np.zeros((N, T_, H))

        h = np.dot(h_prev, self.Wh) + np.dot(x[:,0,:], self.Wx) + self.b
        h = np.tanh(h)
        self.h[:,0,:] = h.copy()

        for t in range(1, T_):
            h = np.dot(h, self.Wh) + np.dot(x[:,t,:], self.Wx) + self.b
            h = np.tanh(h)
            self.h[:,t,:] = h

        return self.h

    def backward(self, dh_out):
        """
        dh_out shape : (N, T, H)
        dx shape : (N, T, D)

        self.x shape : (N, T, D)
        self.h shape : (N, T, H)
        self.Wx shape : (D, H)
        self.Wh shape : (H, H)
        """
        N, T_, H = dh_out.shape
        self.dWh = np.zeros_like(self.Wh).astype('f')
        self.dWx = np.zeros_like(self.Wx).astype('f')
        self.db = np.zeros_like(self.b).astype('f')
        dx = np.zeros_like(self.x).astype('f')

        dh = 0.0
        for t in reversed(range(1, T_)):
            dt = (1 - self.h[:, t, :]**2) * (dh_out[:, t, :] + dh) # dt shape : (N, H)
            self.dWx += np.matmul(self.x[:, t, :].T, dt) 
            self.dWh += np.matmul(self.h[:, t-1, :].T, dt)
            self.db += np.sum(dt, axis=0)

            dx[:, t, :] = np.matmul(dt, self.Wx.T)
            dh = np.matmul(dt, self.Wh.T)
        
        dt = (1 - self.h[:, 0, :]**2) * (dh_out[:, 0, :] + dh)
        self.dWx += np.matmul(self.x[:, 0, :].T, dt)
        self.dWh += np.matmul(self.h_prev.T, dt)
        self.db += np.sum(dt, axis=0)

        dx[:, 0, :] = np.matmul(dt, self.Wx.T)

        return dx


class TimeLSTM:
    def __init__(self, Wh, Wx, b):
        """
        self.D, self.H = Wxi.shape
        self.Wh = np.hstack((Whi, Whf, Who, Whg))
        self.Wx = np.hstack((Wxi, Wxf, Wxo, Wxg))
        self.b = np.concatenate((bi, bf, bo, bg), axis=None)
        """
        self.D = Wx.shape[0]
        self.H = Wx.shape[1] * 0.25
        self.Wh = Wh
        self.Wx = Wx
        self.b = b

        self.dWh = None
        self.dWx = None
        self.db = None

        self.i, self.o, self.f, self.g = None, None, None, None
        self.h, self.c = None, None
        self.c_prev, self.h_prev, self.x = None, None, None
        self.tanh_c = None

    def forward(self, x, h_prev, c_prev):
        """
        x shape : (N, T, D)
        output shape : (N, T, H)
        """
        N, T_, D = x.shape
        H = self.Wh.shape[0]

        self.x = x
        if h_prev is None:
            h_prev = np.zeros((N, H)).astype('f')
        if c_prev is None:
            c_prev = np.zeros((N, H)).astype('f')
        self.h_prev = h_prev
        self.c_prev = c_prev
        self.h = np.zeros((N, T_, H)).astype('f')
        self.c = np.zeros((N, T_, H)).astype('f')
        self.tanh_c = np.zeros((N,T_,H)).astype('f')

        self.i = np.zeros((N, T_, H)).astype('f')
        self.f = np.zeros((N, T_, H)).astype('f')
        self.o = np.zeros((N, T_, H)).astype('f')
        self.g = np.zeros((N, T_, H)).astype('f')

        # first time step
        xw = np.matmul(x[:,0,:], self.Wx)
        hw = np.matmul(h_prev, self.Wh)

        sum_t = xw + hw + self.b

        # slice sum_t for each gate
        sum_f = sum_t[:,    :H] # 0 - H-1
        sum_g = sum_t[:,   H:2*H] # H - 2H-1
        sum_i = sum_t[:, 2*H:3*H] # 2H - 3H-1
        sum_o = sum_t[:, 3*H:]

        self.f[:,0,:] = sigmoid(sum_f)
        self.g[:,0,:] = np.tanh(sum_g)
        self.i[:,0,:] = sigmoid(sum_i)
        self.o[:,0,:] = sigmoid(sum_o)

        c_next = self.f[:,0,:] * c_prev + self.g[:,0,:] * self.i[:,0,:]
        self.tanh_c[:,0,:] = np.tanh(c_next)
        h_next = self.o[:,0,:] * self.tanh_c[:,0,:]

        self.h[:,0,:] = h_next.copy()
        self.c[:,0,:] = c_next.copy()

        # timesteps
        for t in range(1, T_):
            xw = np.matmul(x[:,t,:], self.Wx)
            hw = np.matmul(self.h[:,t-1,:], self.Wh)

            sum_t = xw + hw + self.b

            # slice sum_t for each gate
            sum_f = sum_t[:,    :H] # 0 - H-1
            sum_g = sum_t[:,   H:2*H] # H - 2H-1
            sum_i = sum_t[:, 2*H:3*H] # 2H - 3H-1
            sum_o = sum_t[:, 3*H:]

            self.f[:,t,:] = sigmoid(sum_f)
            self.g[:,t,:] = np.tanh(sum_g)
            self.i[:,t,:] = sigmoid(sum_i)
            self.o[:,t,:] = sigmoid(sum_o)

            self.c[:,t,:] = self.f[:,t,:] * self.c[:,t-1,:] + self.g[:,t,:] * self.i[:,t,:]
            self.tanh_c[:,t,:] = np.tanh(self.c[:,t,:])
            self.h[:,t,:] = self.o[:,t,:] * self.tanh_c[:,t,:]


        return self.h, self.c[:,-1,:]

    def backward(self, dh_out):
        """
        dh_out shape : (N, T, H)
        dx shape : (N, T, D)

        self.x shape : (N, T, D)
        self.h shape : (N, T, H)
        self.Wx shape : (D, H)
        self.Wh shape : (H, H)
        """

        N, T_, H = dh_out.shape
        self.dWh = np.zeros_like(self.Wh).astype('f')
        self.dWx = np.zeros_like(self.Wx).astype('f')
        self.db = np.zeros_like(self.b).astype('f')
        dx = np.zeros_like(self.x).astype('f')

        dh = 0.0
        dc = 0.0
        for t in reversed(range(1, T_)):
            dtanh_c_next = (dh_out[:,t,:] + dh) * self.o[:,t,:]
            dct = dtanh_c_next * (1 - self.tanh_c[:,t,:]**2)
            dct = dc + dct
            
            di = dct * self.g[:,t,:]
            df = dct * self.c[:,t-1,:]
            do = (dh_out[:,t,:] + dh) * self.tanh_c[:,t,:]
            dg = dct * self.i[:,t,:]

            dc = dct * self.f[:,t,:]
            
            di = di * self.i[:,t,:] * (1 - self.i[:,t,:])
            df = df * self.f[:,t,:] * (1 - self.f[:,t,:])
            do = do * self.o[:,t,:] * (1 - self.o[:,t,:])
            dg = dg * (1 - self.g[:,t,:]**2)

            dsum_t = np.hstack((df, dg, di, do))

            self.db += np.sum(dsum_t, axis=0)
            self.dWx += np.matmul(self.x[:,t,:].T, dsum_t)
            self.dWh += np.matmul(self.h[:,t-1,:].T, dsum_t)
            
            dx[:,t,:] = np.matmul(dsum_t, self.Wx.T)
            dh = np.matmul(dsum_t, self.Wh.T)

        # first node
        dtanh_c_next = (dh_out[:,0,:] + dh) * self.o[:,0,:]
        dct = dtanh_c_next * (1 - self.tanh_c[:,0,:]**2)
        dct = dc + dct
        
        di = dct * self.g[:,0,:]
        df = dct * self.c_prev
        do = (dh_out[:,0,:] + dh) * self.tanh_c[:,0,:]
        dg = dct * self.i[:,0,:]

        dc = dct * self.f[:,0,:]
        
        di = di * self.i[:,0,:] * (1 - self.i[:,0,:])
        df = df * self.f[:,0,:] * (1 - self.f[:,0,:])
        do = do * self.o[:,0,:] * (1 - self.o[:,0,:])
        dg = dg * (1 - self.g[:,0,:]**2)

        dsum_t = np.hstack((df, dg, di, do))

        self.db += np.sum(dsum_t, axis=0)
        self.dWx += np.matmul(self.x[:,0,:].T, dsum_t)
        self.dWh += np.matmul(self.h_prev.T, dsum_t)
        
        dx[:,0,:] = np.matmul(dsum_t, self.Wx.T)
        dh = np.matmul(dsum_t, self.Wh.T)

        return dx

class TimeAffine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.dW = None
        self.db = None
        self.x = None

    def forward(self, x):
        """
        x shape : (N, T, D)
        self.W shape : (D, H)
        self.b shape : (H,)
        output shape : (N, T, H)
        """
        N, T_, D = x.shape
        self.x = x.reshape(N * T_, D)

        output = np.matmul(self.x, self.W) + self.b
        output = output.reshape(N, T_, -1)
        return output

    def backward(self, dout):
        """
        dout shape : (N, T, H)
        """
        N, T_, H = dout.shape
        dout = dout.reshape(-1, H)
        self.dW = np.matmul(self.x.T, dout)
        dx = np.matmul(dout, self.W.T)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(N, T_, self.W.shape[0])
        return dx


class TimeEmbedding:
    def __init__(self, W):
        self.W = W
        self.idx = None
        self.dW = None

    def forward(self, idx):
        """
        idx shape : (N, T,) : labels
                    (N, T, V) : one-hot vec
        output shape : (N, T, D)
        """
        if idx.ndim == 3:
            idx = np.argmax(idx, axis=2)

        self.idx = idx
        out = self.W[idx]

        return out

    def backward(self, dout):
        """
        dout shape : (N, T, D)
        """
        D = dout.shape[2]

        idx = self.idx.flatten()
        dout = dout.reshape(-1, D)
        
        self.dW = np.zeros_like(self.W)
        for i, ix in enumerate(idx): 
            self.dW[ix] += dout[i]

        return
    

class TimeSoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.t = None
        self.y = None
        self.ignore_label = -1
        self.mask = None
        self.N, self.T_ = None, None

    def forward(self, x, t):
        """
        x shape : (N, T, D)
        t shape : (N, T)
                  (N, T, V)
        """
        N, T_, D = x.shape
        if t.ndim==3:
            t = np.argmax(t, axis=2)

        self.mask = (t == self.ignore_label)
        self.mask = self.mask.reshape(N * T_)

        self.t = t.reshape(N * T_)
        x = x.reshape(N * T_, D)
        
        self.y = softmax(x)

        ls = np.log(self.y[np.arange(N * T_), self.t] + 1e-7)
        self.loss = -np.sum(ls) 
        self.loss /= ((N * T_) - self.mask.sum())
        #self.loss = cross_entropy_error(self.y.reshape(N*T, C), t)

        self.N = N
        self.T_ = T_
        return self.loss

    def backward(self, dout=1):
        NT, D = self.y.shape[0], self.y.shape[1]
        dx = self.y.copy() # (NxT, C)
        dx *= dout

        dx[np.arange(NT), self.t[np.arange(NT)]] -= 1.0
        dx[self.mask[np.arange(NT)]] = 0.0

        dx /= ((NT) - self.mask.sum())

        dx = dx.reshape(self.N, self.T_, -1)
        return dx
    
class TimeSoftmaxWithLoss_manytoone:
    def __init__(self):
        self.loss = None
        self.t = None
        self.y = None
        self.N, self.T_ = None, None

    def forward(self, x, t):
        """
        x shape : (N, T, D)
        t shape : (N, )
        """
        N, T_, D = x.shape

        self.t = t # (N,)
        x = x[:, -1, :]
        
        self.y = softmax(x) # (N, D)

        ls = np.log(self.y[np.arange(N), self.t] + 1e-7)
        self.loss = -np.sum(ls)
        self.loss /= N
        #self.loss = cross_entropy_error(self.y.reshape(N*T, C), t)

        self.N = N
        self.T_ = T_
        return self.loss

    def backward(self, dout=1):
        N, D = self.y.shape[0], self.y.shape[1]
        dx = np.zeros(shape=(N, self.T_, D))
        dx[:, -1, :] = self.y
        dx *= dout

        dx[np.arange(N), -1, self.t[np.arange(N)]] -= 1.0

        dx /= N
        return dx

if __name__=='__main__':
    layer = TimeSoftmaxWithLoss()
    x = np.random.randint