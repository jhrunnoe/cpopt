#
# @file   : Placer.py
# @author : Jeb Runnoe
# @date   : Feb 2023
# @brief  : Cell placement optimization infrastructure class
#

import numpy as np
from scipy.fft import idct, idst, dctn, idctn
import scipy.sparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from Design import Design
from Net import Net
np.set_printoptions(edgeitems=30, linewidth=100000)

class Placer:
  mu  = 1
  target_density = 0
  smoothing_param = 256
  resolution = 512
  pad_factor = 0.10
  exact = False

  def __init__(self, name):
    self.design = Design(name)
    self.design.extract()

    min_dim = min(self.design.R['dx'], self.design.R['dy'])
    self.grid = dict()
    self.grid['n']   = {'x': int(np.ceil(self.resolution*self.design.R['dx']/min_dim)), 
                        'y': int(np.ceil(self.resolution*self.design.R['dy']/min_dim))}
    self.grid['pad'] = {'x': int(np.ceil(self.pad_factor*self.grid['n']['x'])),
                        'y': int(np.ceil(self.pad_factor*self.grid['n']['y']))}
    self.grid['all'] = {'x': self.grid['n']['x'] + 2*self.grid['pad']['x'],
                        'y': self.grid['n']['y'] + 2*self.grid['pad']['y']}    
    
    self.iofx = lambda x: x + self.grid['pad']['x']
    self.jofy = lambda y: y + self.grid['pad']['y']
    self.xofi = lambda i: i - self.grid['pad']['x']
    self.yofj = lambda j: j - self.grid['pad']['y']

    # Rescale the design so that the grid elements are 1x1
    self.scale = {'x': self.grid['n']['x']/self.design.R['dx'],
                  'y': self.grid['n']['y']/self.design.R['dy']}
    
    self.design.R['x']  *= self.scale['x']
    self.design.R['y']  *= self.scale['y']
    self.design.R['dx'] *= self.scale['x']
    self.design.R['dy'] *= self.scale['y']
    self.design.x0 *= self.scale['x']
    self.design.y0 *= self.scale['y']
    self.design.z0 = np.concatenate((self.design.x0, self.design.y0))
    self.design.dx *= self.scale['x']
    self.design.dy *= self.scale['y']
    self.design.dz = np.concatenate((self.design.dx, self.design.dy))
    self.design.px *= self.scale['x'] 
    self.design.py *= self.scale['y']
    self.design.pz = np.concatenate((self.design.px, self.design.py))

    # convenience definitions
    self.ix = range(self.design.n_cells)
    self.iy = range(self.design.n_cells, 2*self.design.n_cells)
    self.half_dx = np.divide(self.design.dx, 2)
    self.half_dy = np.divide(self.design.dy, 2)
    self.half_dz = np.divide(self.design.dz, 2)

    self.nets = np.ndarray(self.design.n_nets, dtype=object)
    for j in range(self.design.n_nets):
      self.nets[j] = Net(j, self.design.netlist[j], self.design.px, self.design.py)

    # bound constraints
    self.lx = self.half_dx
    self.ly = self.half_dy
    self.lz = self.half_dz
    self.ux = self.design.R['dx']*np.ones_like(self.half_dx) - self.half_dx
    self.uy = self.design.R['dy']*np.ones_like(self.half_dy) - self.half_dy
    self.uz = np.concatenate((self.ux, self.uy))

    # electrostatic quatities: 
    (self.density, self.masks) = self.compute_density(self.design.x0, self.design.y0) # charge density map
    self.potential = np.empty((self.grid['all']['x'], self.grid['all']['y']))         # electrostatic potential map
    self.field     = {'x': np.empty((self.grid['all']['x'], self.grid['all']['y'])),  # electrostatic x & y component field maps
                      'y': np.empty((self.grid['all']['x'], self.grid['all']['y']))} 
            
  def evaluate(self, z):
    x = np.median((self.lx - self.grid['pad']['x'], z[self.ix], self.ux + self.grid['pad']['x']), axis=0)
    y = np.median((self.ly - self.grid['pad']['y'], z[self.iy], self.uy + self.grid['pad']['y']), axis=0)

    (D, M)    = self.compute_density(x, y)
    [P, F, E] = self.compute_electrostatics(D)

    # Determine the grid element(s) in which each cell locates, extract the
    # electric field at those locations, and scale by the cell charges (area). 
    dE = {'x': np.zeros_like(x), 'y': np.zeros_like(y)}

    for k in range(self.design.n_cells):
      [I, J] = self.masks[k].nonzero()
      if self.exact:
        dE['x'][k] = -np.multiply(F['x'][I, J], M[k][I, J]).sum()
        dE['y'][k] = -np.multiply(F['y'][I, J], M[k][I, J]).sum()
      else:
        dE['x'][k] = -F['x'][I, J].sum()
        dE['y'][k] = -F['y'][I, J].sum()

    W    = 0.0
    dW = {'x': np.zeros_like(x), 'y': np.zeros_like(y)}

    for j in range(self.design.n_nets):
      (nW, ndW) = self.nets[j].pin_variance(x, y)
      W += nW
      dW['x'][self.nets[j].cells] += ndW['x']
      dW['y'][self.nets[j].cells] += ndW['y']

    df = {'x': dW['x'] + self.mu*dE['x'], 
          'y': dW['y'] + self.mu*dE['y']}
    f  =  W + self.mu*E
    g  = np.concatenate((df['x'], df['y']))
    return(f, g)

  def update_density(self, x, y, cells, D, M):
    '''
    @brief: Update the density map by subtracting, shifting, and adding density masks of provided cells
    '''
    for k in cells:
      di = int(np.floor(self.iofx(x[k] - self.half_dx[k]))) - M[k].nonzero()[0][0] # Change in row index
      dj = int(np.floor(self.jofy(y[k] - self.half_dy[k]))) - M[k].nonzero()[1][0] # Change in column index

      if di != 0 or dj != 0: 
        D = D._add_sparse(-M[k]) # Subtract out this cells previous density contribution

        indices = M[k].indices + dj # Shift column indices 
    
        # Shift row index pointers (in CSR format)
        # For example,
        # shift   up 3 rows:  [0, 0, 0, 0, 4, 9, 13, 18, 18, 18, 18, 18] -> [0, 4, 9, 13, 18, 18, 18, 18, 18, 18, 18, 18]
        # shift down 2 rows:  [0, 0, 0, 0, 4, 9, 13, 18, 18, 18, 18, 18] -> [0, 0, 0,  0,  0,  0,  4,  9, 13, 18, 18, 18]
        if di > 0:
          pointers = np.concatenate((np.zeros(di), M[k].indptr[:-di])).astype(int)
        elif di < 0: 
          pointers = np.concatenate((M[k].indptr[abs(di):], M[k].nnz*np.ones(abs(di)))).astype(int)
        else: 
          pointers = M[k].indptr
        
        M[k] = scipy.sparse.csr_matrix((M[k].data, indices, pointers), shape=(self.grid['all']['x'], self.grid['all']['y']))

        D = D._add_sparse(M[k])
    D.data = np.maximum(D.data - self.target_density, 0)
    return (D, M)

  def compute_density(self, x, y):
    '''
    @brief   : computation of the cell density map based on the current placement
    @param x : current placement x-coordinates
    @param y : current placement y-coordinates
    '''
    D = scipy.sparse.csr_matrix((self.grid['all']['x'], self.grid['all']['y']))
    M = [None]*self.design.n_cells

    for k in range(self.design.n_cells):
      lx = int(np.floor(x[k] - self.half_dx[k]))
      ly = int(np.floor(y[k] - self.half_dy[k]))
      ux = int(np.floor(x[k] + self.half_dx[k]))
      uy = int(np.floor(y[k] + self.half_dy[k]))
      dx_k = ux - lx
      dy_k = uy - ly

      if self.exact:
        block = np.zeros((dx_k + 1, dy_k + 1))      

        # Width and height of partial overlaps
        ldx = min((lx + 1) - (x[k] - self.half_dx[k]), self.design.dx[k]) # lower horizontal
        ldy = min((ly + 1) - (y[k] - self.half_dy[k]), self.design.dy[k]) # lower vertical
        udx = (x[k] + self.half_dx[k]) - ux if dx_k > 0 else 0 # upper horizontal
        udy = (y[k] + self.half_dy[k]) - uy if dy_k > 0 else 0 # upper vertical

        block[0,           0] += ldx*ldy # lower left 
        block[0,        dy_k] += ldx*udy # upper left 
        block[  dx_k,      0] += udx*ldy # lower right
        block[  dx_k,   dy_k] += udx*udy # upper right
        block[1:dx_k,      0] += ldy     # lower horizontal strip
        block[0,      1:dy_k] += ldx     # left vertical strip
        block[  dx_k, 1:dy_k] += udx     # right vertical strip
        block[1:dx_k,   dy_k] += udy     # upper horizontal strip
        block[1:dx_k, 1:dy_k] += 1.0     # interior
      else:
        block = np.ones((dx_k + 1, dy_k + 1))

      [I, J] = np.meshgrid(self.iofx(np.arange(lx, ux + 1)), self.jofy(np.arange(ly, uy + 1)), indexing='ij')
      M[k]   = scipy.sparse.csr_matrix((block.ravel(), (I.ravel(), J.ravel())), shape=(self.grid['all']['x'], self.grid['all']['y']))
      D      = D._add_sparse(M[k])
    D.data = np.maximum(D.data - self.target_density, 0)
    return(D, M)

  def compute_electrostatics(self, D):
    '''
    @brief : Computation of the system electrostatic quantities determined by the density map
    @param : D is a matrix with the density value at each grid element 
    '''
    if scipy.sparse.issparse(D):
      D = D.todense() # DFT routines don't work with sparse arrays
      
    (M, N) = D.shape

    # Define the frequency mesh for the cosine bases
    u = (np.pi/M)*np.arange(M)
    v = (np.pi/N)*np.arange(N)

    # compute the Fourier coefficients to express the density map D in the cosine basis 
    # over the u-v frequency mesh.
    A = dctn(D, type=2, norm="ortho")

    # the electrostatic potential map is obtained by integrating the cosine representation
    # of the density map over the placement region
    S       = np.add.outer(np.square(u), np.square(v)) # (S)um of squares matrix
    S[0, 0] = 1
    C       = np.divide(A, S)
    C[0, 0] = 0
    P       = idctn(C, type=2, norm="ortho")

    F = dict.fromkeys(['x', 'y']) 
    # Differentiating the potential map with respect to x gives the 
    # horizontal component of the electrostatic field. This essentially multiplies
    # each row of the coefficient matrix C by elements of u
    Cu = np.multiply(C.T, u).T    
    F['x'] = idst(np.vstack((idct(Cu, axis=1, norm="ortho")[1:, :], np.zeros([1, N]))), axis=0, norm="ortho")
     
    # Differentiating the potential map with respect to y gives the vertical
    # component of the electrostatic field. This is done by multiplying the
    # indices of C by elements of v.
    Cv = np.multiply(C, v)
    F['y'] = idct(idst(np.hstack((Cv[:, 1:], np.zeros([M, 1]))), axis=1, norm="ortho"), axis=0, norm="ortho")

    E = 0.50*np.multiply(P, D).sum()    
    return(P, F, E)

  def plot_cells(self, x, y, cells="all"):
    plt.xlim(-1.25*self.grid['pad']['x'], self.grid['n']['x'] + 1.25*self.grid['pad']['x'])
    plt.ylim(-1.25*self.grid['pad']['y'], self.grid['n']['y'] + 1.25*self.grid['pad']['y'])
    plt.axhline(0, color='black', linewidth=.5)
    plt.axvline(0, color='black', linewidth=.5)

    if cells == "all":
      cells = range(self.design.n_cells)

    # Show the entire padded region
    plt.gca().add_patch(Rectangle((-self.grid['pad']['x'], -self.grid['pad']['y']), 
                                    self.grid['n']['x'] + 2*self.grid['pad']['x'],
                                    self.grid['n']['y'] + 2*self.grid['pad']['y'],
                                    edgecolor='red', facecolor='none', alpha=1, lw=1))
    
    # Show the actual placement region
    plt.gca().add_patch(Rectangle((0, 0), self.design.R['dx'], self.design.R['dy'], 
                                   edgecolor='green', facecolor=[0, 1, 0, .05], ls='--'))

    for i in cells:
      plt.gca().add_patch(Rectangle((x[i] - self.half_dx[i], y[i] - self.half_dy[i]), 
                                     self.design.dx[i], self.design.dy[i], 
                                     edgecolor='black', facecolor='none', alpha=1))

  def plot_heatmap(self, Z):
    plt.xlim(-1.25*self.grid['pad']['x'], self.grid['n']['x'] + 1.25*self.grid['pad']['x'])
    plt.ylim(-1.25*self.grid['pad']['y'], self.grid['n']['y'] + 1.25*self.grid['pad']['y'])
    plt.axhline(0, color='black', linewidth=.5)
    plt.axvline(0, color='black', linewidth=.5)
    X, Y = np.meshgrid(self.xofi(np.arange(Z.shape[0])), self.yofj(np.arange(Z.shape[1])), indexing='ij')
    hm = plt.gca().pcolormesh(X, Y, Z)
    plt.colorbar(hm)

  def plot_contour(self, Z):
    plt.xlim(-1.25*self.grid['pad']['x'], self.grid['n']['x'] + 1.25*self.grid['pad']['x'])
    plt.ylim(-1.25*self.grid['pad']['y'], self.grid['n']['y'] + 1.25*self.grid['pad']['y'])
    plt.axhline(0, color='black', linewidth=.5)
    plt.axvline(0, color='black', linewidth=.5)
    X, Y = np.meshgrid(self.xofi(np.arange(Z.shape[0])), self.yofj(np.arange(Z.shape[1])), indexing='ij')
    plt.gca().contour(X, Y, Z, 30)

  def plot_surface(self, Z):
    X, Y = np.meshgrid(self.xofi(np.arange(Z.shape[0])), self.yofj(np.arange(Z.shape[1])), indexing='ij')
    plt.gca(projection='3d').plot_surface(X, Y, Z)

  @staticmethod
  def idsxt(X, axis=1):
    if axis == 0:
      X = X.T
    [m, n] = X.shape
    Y = np.zeros_like(X)
    for i in range(m):
      y = idct(np.flip(np.append(X[i, 1:], 0)), norm="ortho")
      y[1::2] = -y[1::2]
      Y[i,:]  = y
    if axis == 0:
      Y = Y.T
    return Y

  @staticmethod
  def overlap(r1, r2):
    ''' 
    Computes the area of the intersection of two rectangles r1 and r2
    '''
    dx = max(0, min(r1['x'] + 0.50*r1['dx'], r2['x'] + 0.50*r2['dx']) - max(r1['x'] - 0.50*r1['dx'], r2['x'] - 0.50*r2['dx']))
    dy = max(0, min(r1['y'] + 0.50*r1['dy'], r2['y'] + 0.50*r2['dy']) - max(r1['y'] - 0.50*r1['dy'], r2['y'] - 0.50*r2['dy']))
    return(dx*dy)