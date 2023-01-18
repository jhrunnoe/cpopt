import numpy as np
from scipy.fft import idct, idst, dctn, idctn
import matplotlib.pyplot as plt
from Cell import Cell
from Design import Design
from Net import Net
from Rect import Rect

class Placer:
  # the number of horizontal and vertical grid elements
  ngx = 64
  ngy = 64

  mu  = 1
  
  def __init__(self, name):
    design = Design(name)
    assert design.R['x'] == 0 and design.R['y'] == 0, "The placement region must be anchored at the origin."

    self.R = Rect(0, 0, design.R['dx'], design.R['dy'])
    self.nCells = design.nCells
    self.nNets  = design.nNets
    
    self.x  = design.x0 + 0.70*(self.R.cx - design.x0)
    self.y  = design.y0 + 0.70*(self.R.cy - design.y0)
    self.dx = design.dx
    self.dy = design.dy

    self.C = np.empty(self.nCells, dtype = object)
    for i in range(self.nCells):
      self.C[i] = Cell(self, i)

    self.N = np.empty(self.nNets, dtype = object)
    for i in range(self.nNets):
      self.N[i] = Net(self, design.N[i])

    # Compute the width and height of the grid elements
    self.dgx = round(self.R.dx/self.ngx)
    self.dgy = round(self.R.dy/self.ngy)
    
    # Each element in the ngx-by-ngy grid is associated with the electrostatic quatities: 
    self.density   = np.empty((self.ngx, self.ngy))    # - The (charge) density rho(x, y)
    self.potential = np.empty((self.ngx, self.ngy))    # - The electrostatic potential psi(x, y)
    self.force     = np.empty((2, self.ngx, self.ngy)) # - The electrostatic force xi(x, y) = (xi_x(x, y), xi_y(x, y))

  def Overlap(self, r1, r2):
    ''' 
    Computes the area of the intersection of two rectangles r1 and r2
    '''
    dx = max(0, min(r1.x + r1.dx, r2.x + r2.dx) - max(r1.x, r2.x))
    dy = max(0, min(r1.y + r1.dy, r2.y + r2.dy) - max(r1.y, r2.y))
    return dx*dy

  def Objective(self, x, y):
    self.x = x
    self.y = y
    self.ComputeDensity() # Compute the current density map
    self.ComputeES()
    W = sum([net.var for net in self.N])

    return W + self.mu*self.energy
    

  def ComputeDensity(self):
    '''
      ComputeDensity() determines the density map based on the current placement
    '''
    self.density.fill(0.0) 

    for c in range(self.nCells):
      li = int(np.floor(self.x[c]/self.dgx))
      lj = int(np.floor(self.y[c]/self.dgy))
      ui = int(np.floor((self.x[c] + self.dx[c])/self.dgx))
      uj = int(np.floor((self.y[c] + self.dy[c])/self.dgy))
      for i in range(li, ui):
        for j in range(lj, uj):
          overlap =  self.Overlap(self.C[c], Rect(i*self.dgx, j*self.dgy, self.dgx, self.dgy))
          self.density[i, j] += overlap/(self.dgx*self.dgy)
  
  def ComputeES(self):
    '''
      ComputeES() computes the electrostatics of the system using discrete cosine and sine
      transforms and their inverses. 
    '''

    # Define the frequencies for the cosine bases
    u = (np.pi/self.ngx)*(np.arange(self.ngx) + 0.50)
    v = (np.pi/self.ngy)*(np.arange(self.ngy) + 0.50)

    # Compute the coefficient matrices 
    SS = np.add.outer(np.square(u), np.square(v)) # (S)um of (S)quares
    C  = dctn(self.density, type = 4) 
    C  = np.divide(C, SS)
    Cu = np.multiply(C.T, u).T
    Cv = np.multiply(C, v)

    # Compute the system electrostatics
    self.potential   = idctn(C, type = 4)
    self.force[0, :] = idst(idct(Cu, type = 4), type = 4, axis = 0)
    self.force[1, :] = idct(idst(Cv, type = 4), type = 4, axis = 0)
    self.energy      = np.multiply(self.potential, self.density).sum()

  def PlotHeatmap(self):
    X, Y = np.meshgrid(np.arange(self.ngx), np.arange(self.ngy), indexing='ij')
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].pcolormesh(X, Y, self.density)
    axes[0, 1].pcolormesh(X, Y, self.potential)
    axes[1, 0].pcolormesh(X, Y, self.force[0, :])
    axes[1, 1].pcolormesh(X, Y, self.force[1, :])
    plt.tight_layout()
    plt.show()           

  def PlotContour(self):
    X, Y = np.meshgrid(np.arange(self.ngx), np.arange(self.ngy), indexing='ij')
    fig, ax = plt.subplots()
    ax.contour(X, Y, self.potential, 30)
    ax.quiver(X, Y, self.force[0, :], self.force[1, :])
    plt.tight_layout()
    plt.show()

  def PlotSurface(self):
    X, Y = np.meshgrid(np.arange(self.ngx), np.arange(self.ngy), indexing='ij')
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1, projection = '3d')
    ax.plot_surface(X, Y, self.density)
    ax = fig.add_subplot(2, 2, 2, projection = '3d')
    ax.plot_surface(X, Y, self.potential)
    ax = fig.add_subplot(2, 2, 3, projection = '3d')
    ax.plot_surface(X, Y, self.force[0, :])
    ax = fig.add_subplot(2, 2, 4, projection = '3d')
    ax.plot_surface(X, Y, self.force[1, :])
    plt.tight_layout()
    plt.show()
    