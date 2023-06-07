import numpy as np

class Net:
    smoothing = 256
    def __init__(self, ID, cells, px, py):
        self.ID    = ID
        self.cells = cells
        self.size  = len(cells)
        self.px    = px
        self.py    = py
  
    def locate_pins(self, x, y):
        pins = {'x': x[self.cells] + self.px[self.cells], 
                'y': y[self.cells] + self.py[self.cells]}
        return(pins)
    
    def WAWL(self, x, y):
        '''
        @brief         : Computation of weighted average smooth approximation of the net's half-perimeter wirelength
        @param x       : current placement x coordinates
        @param y       : current placement y coordinates
        @param cellIDs : IDs of cells in the given net
        '''
        pins = self.locate_pins(x, y)
        (Smaxx, dSmaxx) = self.smooth_extremum(pins['x'],  self.smoothing)
        (Smaxy, dSmaxy) = self.smooth_extremum(pins['y'],  self.smoothing)
        (Sminx, dSminx) = self.smooth_extremum(pins['x'], -self.smoothing)
        (Sminy, dSminy) = self.smooth_extremum(pins['y'], -self.smoothing)
        W = (Smaxx - Sminx) + (Smaxy - Sminy)
        dW = {'x': dSmaxx - dSminx, 
              'y': dSmaxy - dSminy}
        return(W, dW)
    
    def pin_variance(self, x, y):
        pins = self.locate_pins(x, y)
        V = np.var(pins['x']) + np.var(pins['y'])
        dV = {'x': (2/self.size)*(pins['x'] - np.mean(pins['x'])), 
              'y': (2/self.size)*(pins['y'] - np.mean(pins['y']))}
        return(V, dV)
    
    def HPWL(self, x, y):
        pins = self.locate_pins(x, y)
        hpwl = (pins['x'].max() - pins['x'].min()) + (pins['y'].max() - pins['y'].min())
        return(hpwl)
    
    @staticmethod
    def smooth_extremum(z, t):
        '''
        @brief normalized weighted average smooth approximation of maximum or minimum
        @param z : any vector (assumed z.shape = (len(z),))
        @param t : the smoothing parameter (t>0: maximum, t<0: minimum)
        '''
        z_norm = np.linalg.norm(z, 2)

        u = z/z_norm
        v = np.exp(t*u)
        w = v/v.sum()

        S = np.dot(w, u)
        q = np.multiply(w, 1 + t*(u - S))

        S_  = z_norm*S
        dS_ = q + (S - np.dot(q, u))*u
        return(S_, dS_)