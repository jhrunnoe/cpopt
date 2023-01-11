import numpy as np
from design import Design
from scipy.fft import idct, idst, dctn, idctn, dstn, idstn
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Placer:
  # Specify the desired number of horizontal and vertical grid elements
  ngx = 32
  ngy = 32

  DENS_IDX = 0
  POTN_IDX = 1
  FRCX_IDX = 2
  FRCY_IDX = 3

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
      c.px = c.cx # using midpoint as pin location for now
      c.py = c.cy
      c.ID = ID
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

    # Compute the width and height of the grid elements
    self.dgx = round(self.R.dx/self.ngx)
    self.dgy = round(self.R.dy/self.ngy)

    # Grid initialization:
    # Each element in the ngx-by-ngy grid has four associated quantities
    # G[0, i, j] - density at (i, j) 
    # G[1, i, j] - potential at (i, j)
    # G[2, i, j] - field x component at (i, j)
    # G[3, i, j] - field y component at (i, j)
    self.G = np.empty((4, self.ngx, self.ngy))
    self.density = np.empty((self.ngx, self.ngy))
    self.potential = np.empty((self.ngx, self.ngy))
    self.xForce = np.empty((self.ngx, self.ngy))
    self.yForce = np.empty((self.ngx, self.ngy))

  def Overlap(self, r1, r2):
    ''' 
    Computes the area of the intersection of two rectangles r1 and r2
    '''
    dx = max(0, min(r1.x + r1.dx, r2.x + r2.dx) - max(r1.x, r2.x))
    dy = max(0, min(r1.y + r1.dy, r2.y + r2.dy) - max(r1.y, r2.y))
    return dx*dy

  def Density(self):
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

  
  def FFT(self):
    self.Density() # Compute the current density map

    # First the coefficients must be computed:
    # Define the frequencies for the trig bases
    wu = (2*np.pi/self.ngx)*np.arange(self.ngx)
    wv = (2*np.pi/self.ngy)*np.arange(self.ngy)

    # The (u, v) entry of this matrix is w[u]^2 + w[v]^2
    wu2_plus_wv2 = np.add.outer(np.square(wu), np.square(wv))

    wu2_plus_wv2[0, 0] = 1.0 # To avoid division by zero
    inv_wu2_plus_wv2 = 1.0/wu2_plus_wv2 
    inv_wu2_plus_wv2[0, 0] = 0.0

    # This multiplies the u-th row by w[u]:
    wu_by_wu2_plus_wv2 = np.multiply(inv_wu2_plus_wv2.T, wu).T
    # This multiplies the v-th column by w[v]
    wv_by_wu2_plus_wv2 = np.multiply(inv_wu2_plus_wv2, wv)

    auv = dctn(self.density)

    auv_wu_by_wu2_plus_wv2 = np.multiply(auv, wu_by_wu2_plus_wv2)
    auv_wv_by_wu2_plus_wv2 = np.multiply(auv, wv_by_wu2_plus_wv2)
    auv_by_wu2_plus_wv2    = np.multiply(auv, inv_wu2_plus_wv2)

    self.potential = idctn(auv_by_wu2_plus_wv2)
    self.energy = np.multiply(self.potential, self.density).sum()

    a = idct(auv_wu_by_wu2_plus_wv2)
    self.xForce = idst(a, axis = 0)
    b = idst(auv_wv_by_wu2_plus_wv2)
    self.yForce = idct(b, axis = 0)      

    X, Y = np.meshgrid(np.arange(self.ngx), np.arange(self.ngy), indexing='ij')

    fig, axes = plt.subplots(2, 2)
    axes[0, 0].pcolormesh(X, Y, self.density)
    axes[0, 1].pcolormesh(X, Y, self.potential)
    axes[0, 1].quiver(X, Y, self.xForce, self.yForce)
    axes[1, 0].pcolormesh(X, Y, self.xForce)
    axes[1, 1].pcolormesh(X, Y, self.yForce)
    plt.tight_layout()

    # plt.xlim([- 0.05*self.R.dx, 1.05*self.R.dx])
    # plt.ylim([- 0.05*self.R.dy, 1.05*self.R.dy])
    # for i in range(self.nCells):
    #   axes[0, 0].add_patch(Rectangle((self.x[i], self.y[i]), self.dx[i], self.dy[i],
    #                      edgecolor = 'r',
    #                      linewidth = 1,
    #                      facecolor = 'none'))
    fig, ax = plt.subplots()
    ax.contour(X, Y, self.potential, 20)
    ax.quiver(X, Y, self.xForce, self.yForce)
    plt.tight_layout()
    plt.show()