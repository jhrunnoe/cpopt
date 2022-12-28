import numpy as np
from design import Design
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Rect:
  def __init__(self, x, y, dx, dy):
    self.x  = x
    self.y  = y
    self.dx = dx
    self.dy = dy
    # Derived properties
    self.cx = x + 0.50*dx
    self.cy = y + 0.50*dy
    self.A  = dx*dy

class Bin(Rect):
  def __init__(self, x, y, dx, dy):
    super(Bin, self).__init__(x, y, dx, dy)
    self.density = 0

class Cell(Rect):
  def __init__(self, x, y, dx, dy, ID):
    super(Cell, self).__init__(x, y, dx, dy)
    self.ID = ID
  def getCellx(self, x):
    return x[self.ID]

  def celly(self, y):
    return y[self.ID]

class Net:        
  def __init__(self, cells):
    self.cells = cells
    self.IDs = [c.ID for c in cells]
    self.size = len(self.IDs)
  def getNetx(self, x):
    netx = np.empty((self.size, 1))
    for i in range(self.size):
      netx[i] = self.cells[i].cellx(x)
    return netx
  def getNety(self, y):
    nety = np.empty((self.size, 1))
    for i in range(self.size):
      nety[i] = self.cells[i].celly(y)
    return nety

class Placer:
  gridResolution = 512

  def __init__(self, name):
    chip = Design(name)

    self.nC = chip.nC
    self.nN = chip.nN
    self.R  = Rect(chip.R['x'], chip.R['y'], chip.R['dx'], chip.R['dy'])

    self.x  = chip.x0 + 1*min(self.R.dx - (chip.x0 + chip.dx))*np.random.rand(self.nC)
    self.y  = chip.y0 + 1*min(self.R.dy - (chip.y0 + chip.dy))*np.random.rand(self.nC)
    self.dx = chip.dx
    self.dy = chip.dy

    self.eps = int(np.ceil(min(self.R.dx, self.R.dy))/self.gridResolution)
    self.nGx = int(np.ceil(self.R.dx/self.eps))
    self.nGy = int(np.ceil(self.R.dy/self.eps))

    self.C = np.empty(self.nC, dtype=object)
    for i in range(self.nC):
      self.C[i] = Cell(self.x[i], self.y[i], self.dx[i], self.dy[i], i)

    self.N = np.empty(self.nN, dtype=object)
    for j in range(self.nN):
      self.N[j] = Net(self.C[chip.N[j]])

    self.G = np.empty((self.nGx, self.nGy), dtype=object)
    for i in range(self.nGx):
      for j in range(self.nGy):
        self.G[i, j] = Bin(self.R.x + i*self.eps, self.R.y + j*self.eps, self.eps, self.eps) 

  def Overlap(self, r1, r2):
    dx = max(0, min(r1.x + r1.dx, r2.x + r2.dx) - max(r1.x, r2.x))
    dy = max(0, min(r1.y + r1.dy, r2.y + r2.dy) - max(r1.y, r2.y))
    return dx*dy

  def Density(self, x, y):
    for idx in range(self.nC):
      li = int((x[idx] - np.mod(x[idx], self.eps))/self.eps)
      lj = int((y[idx] - np.mod(y[idx], self.eps))/self.eps)
      ui = int((x[idx] + self.dx[idx] - np.mod(x[idx] + self.dx[idx], self.eps))/self.eps)
      uj = int((y[idx] + self.dy[idx] - np.mod(y[idx] + self.dy[idx], self.eps))/self.eps) 
      
      # Iterate through grid elements intersecting current cell and increment the
      # grid elements' density by (overlap area)/(bin area)

      lx = self.R.x + (li + 1)*self.eps
      ux = self.R.x + ui*self.eps
      ly = self.R.y + (lj + 1)*self.eps
      uy = self.R.y + uj*self.eps

      self.G[li, lj].density += (lx - x[idx])*(ly - y[idx])/self.eps**2
      self.G[li, uj].density += (lx - x[idx])*(y[idx] + self.dy[idx] - uy)/self.eps**2
      self.G[ui, lj].density += (x[idx] + self.dx[idx] - ux)*(ly - y[idx])/self.eps**2
      self.G[ui, uj].density += (x[idx] + self.dx[idx] - ux)*(y[idx] + self.dy[idx] - uy)/self.eps**2
      for j in range(lj + 1, uj):
        self.G[li, j].density += (lx - x[idx])/self.eps
        self.G[ui, j].density += (x[idx] + self.dx[idx] - ux)/self.eps
      for i in range(li + 1, ui):
        self.G[i, lj].density += (ly - y[idx])/self.eps
        self.G[i, uj].density += (y[idx] + self.dy[idx] - uy)/self.eps
      for i in range(li + 1, ui):
        for j in range(lj + 1, uj):
          self.G[i, j].density += 1

  def densityHeatMap(self):
    G = np.zeros((self.nGy, self.nGx))

    for i in range(self.nGy):
      for j in range(self.nGx):
        G[i, j] = self.G[j, (self.nGy - 1) - i].density

    sb.heatmap(G)
    plt.show()