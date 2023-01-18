import numpy as np

class Net:
    def __init__(self, placer, IDs):
      self.placer = placer
      self.IDs = IDs
      self.cells = placer.C[IDs]
      self.size = len(self.IDs)
    
    @property
    def posx(self):
      return self.placer.x[self.IDs]
    
    @property
    def posy(self):
      return self.placer.y[self.IDs]
    
    @property
    def pinx(self):
      return self.posx + 0.50*self.placer.dx[self.IDs]
    
    @property
    def piny(self):
      return self.posy + 0.50*self.placer.dy[self.IDs]
    
    @property
    def hpwl(self):
      dx = np.max(self.pinx) - np.min(self.pinx)
      dy = np.max(self.piny) - np.min(self.piny)
      return 0.50*(dx + dy)
    
    @property
    def var(self):
      vx = np.var(self.pinx)
      vy = np.var(self.piny)
      return 0.50*(vx + vy)

    @property
    def dev(self):
      return np.sqrt(self.var)