import numpy as np
from design import Design
from scipy.fft import idct, idst, dctn, idctn
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Placer:
  # Specify the desired number of horizontal and vertical grid elements
  ngx = 512
  ngy = 512

  class Rect:
    def __init__(r, x, y, dx, dy):
      # Location of bottom left corner
      r.x = x
      r.y = y
      # Width and height
      r.dx = dx
      r.dy = dy
      # Location of center
      r.cx = x + 0.50*dx
      r.cy = y + 0.50*dy

      r.area  = dx*dy 

  class Cell(Rect):
    def __init__(c, placer, ID):
      super().__init__(placer.x[ID], placer.y[ID], placer.dx[ID], placer.dy[ID])
      c.placer = placer
      c.ID = ID
      # using midpoint as pin location for now
      c.px = c.cx 
      c.py = c.cy
      
    @property
    def pinx(c):
      return c.placer.x[c.ID] + 0.50*c.placer.dx[c.ID]
    @property
    def piny(c):
      return c.placer.y[c.ID] + 0.50*c.placer.dy[c.ID]

  class Net:
    def __init__(n, placer, IDs):
      n.placer = placer
      n.IDs = IDs
      n.cells = placer.C[IDs]
      n.size = len(n.IDs)
    
    @property
    def netPinx(n):
      return n.placer.x[n.IDs] + 0.50*n.placer.dx[n.IDs]
    @property
    def netPiny(n):
      return n.placer.y[n.IDs] + 0.50*n.placer.dy[n.IDs]
    @property
    def netHPWL(n):
      dx = np.max(n.netPinx) - np.min(n.netPinx)
      dy = np.max(n.netPiny) - np.min(n.netPiny)
      return 0.50*(dx + dy)
    @property
    def netVar(n):
      vx = np.var(n.netPinx)
      vy = np.var(n.netPiny)
      return 0.50*(vx + vy)

  def __init__(self, name):
    design = Design(name)
    assert design.R['x'] == 0 and design.R['y'] == 0, "The placement region must be anchored at the origin."

    self.R = self.Rect(0, 0, design.R['dx'], design.R['dy'])
    self.nCells = design.nCells
    self.nNets  = design.nNets
    
    self.x  = design.x0 + 0.70*(self.R.cx - design.x0)
    self.y  = design.y0 + 0.70*(self.R.cy - design.y0)
    self.dx = design.dx
    self.dy = design.dy

    self.C = np.empty(self.nCells, dtype = object)
    for i in range(self.nCells):
      self.C[i] = self.Cell(self, i)

    self.N = np.empty(self.nNets, dtype = object)
    for i in range(self.nNets):
      self.N[i] = self.Net(self, design.N[i])

    ### Grid initialization:
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

  def ComputeDensity(self):
    '''
      ComputeDensity() determines the density map based on the current placement
    '''
    self.density.fill(0.0) 

    for c in range(self.nCells):
      li = np.floor(self.x[c]/self.dgx).astype('int')
      lj = np.floor(self.y[c]/self.dgy).astype('int')
      ui = np.floor((self.x[c] + self.dx[c])/self.dgx).astype('int')
      uj = np.floor((self.y[c] + self.dy[c])/self.dgy).astype('int')
      for i in range(li, ui):
        for j in range(lj, uj):
          overlap =  self.Overlap(self.C[c], self.Rect(i*self.dgx, j*self.dgy, self.dgx, self.dgy))
          self.density[i, j] += overlap/(self.dgx*self.dgy)

  
  def ComputeES(self):
    '''
      ComputeES() computes the electrostatics of the system using discrete cosine and sine
      transforms and their inverses. 
    '''
    self.ComputeDensity() # Compute the current density map

    # First the coefficients must be computed:
    # Define the frequencies for the trig bases
    M, N = self.ngx, self.ngy

    u = (np.pi/M)*(np.arange(M) + 0.50)
    v = (np.pi/N)*(np.arange(N) + 0.50)

    C = dctn(self.density, type = 4) 
    
    SS = np.add.outer(np.square(u), np.square(v)) # (S)um of (S)quares

    CbySS = np.divide(C, SS)

    self.potential = idctn(CbySS, type = 4)

    CbySSu = np.multiply(CbySS.T, u).T
    CbySSv = np.multiply(CbySS, v)

    self.force[0, :] = idst(idct(CbySSu, type = 4), type = 4, axis = 0)
    self.force[1, :] = idct(idst(CbySSv, type = 4), type = 4, axis = 0)

    X, Y = np.meshgrid(np.arange(M), np.arange(N), indexing='ij')
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].pcolormesh(X, Y, self.density)
    axes[0, 1].pcolormesh(X, Y, self.potential)
    axes[0, 1].quiver(X, Y, self.force[0, :], self.force[1, :])
    axes[1, 0].pcolormesh(X, Y, self.force[0, :])
    axes[1, 1].pcolormesh(X, Y, self.force[1, :])
    plt.tight_layout()
    plt.show()           
      
    self.energy = np.multiply(self.potential, self.density).sum()

    fig, ax = plt.subplots()
    ax.contour(X, Y, self.potential, 30)
    ax.quiver(X, Y, self.force[0, :], self.force[1, :])
    plt.tight_layout()
    plt.show()