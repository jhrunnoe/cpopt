#
# @file   : Placer.py
# @author : Jeb Runnoe
# @date   : Feb 2023
# @brief  : Cell placement optimization infrastructure class
#

import numpy as np
from scipy.fft import idct, idst, dctn, idctn
from scipy.optimize import Bounds, minimize, BFGS
import scipy.sparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from Design import Design
np.set_printoptions(edgeitems=30, linewidth=100000)

class Placer:
  mu  = 1
  target_density = 1
  smoothing_param = 256
  grid_dim = 512

  def __init__(self, name):
    self.design = Design(name)
    self.design.extract()

    self.grid = dict()
    self.grid['nx']   = self.grid_dim
    self.grid['ny']   = self.grid_dim
    self.grid['n' ]   = self.grid['nx']*self.grid['ny']
    self.grid['dx']   = np.ceil(self.design.R['dx']/self.grid['nx'])
    self.grid['dy']   = np.ceil(self.design.R['dy']/self.grid['ny'])
    self.grid['dxdy'] = self.grid['dx']*self.grid['dy']

    # The number of grid elements each cell occupies horizontally/vertically (not exact)
    self.dgx = np.ceil(np.divide(self.design.dx, self.grid['dx'])) 
    self.dgy = np.ceil(np.divide(self.design.dy, self.grid['dy'])) 

    # electrostatic quatities: 
    self.charge    = np.multiply(self.design.dx, self.design.dy)          # cell charge (area) vector
    [self.density, self.masks] = self.compute_density(self.design.x0, self.design.y0) # charge density map
           
    self.potential = np.empty((self.grid['nx'], self.grid['ny']))         # electrostatic potential map
    self.field     = {'x': np.empty((self.grid['nx'], self.grid['ny'])),  # electrostatic x & y field maps
                      'y': np.empty((self.grid['nx'], self.grid['ny']))} 
    
    self.trace = {      'W': [],       'E': [],       'f': [], 
                  'norm_dW': [], 'norm_dE': [], 'norm_df': []}
    
    # convenience definitions
    self.xind = range(self.design.n_cells)
    self.yind = range(self.design.n_cells, 2*self.design.n_cells)
    self.z0   = np.concatenate((self.design.x0, self.design.y0))
    self.sqzx = self.design.x0 + 0.50*(0.50*self.design.dx - self.design.x0)
    self.sqzy = self.design.y0 + 0.50*(0.50*self.design.dy - self.design.y0)
    self.sqzz = np.append(self.sqzx, self.sqzy)

    self.design.dz = np.append(self.design.dx, self.design.dy)
    self.zk   =np.zeros_like(self.z0) 

    # bound constraints
    self.lz = np.zeros_like(self.z0)
    self.uz = np.concatenate((self.design.R['dx'] - self.design.dx, self.design.R['dy'] - self.design.dy))

  def net_WAWL(self, x, y, cellIDs):
    '''
    @brief         : Computation of weighted average smooth approximation of the half-perimeter
                     wirelength due to the provided cells
    @param x       : current placement x coordinates
    @param y       : current placement y coordinates
    @param cellIDs : IDs of cells in the given net
    '''
    pin_x = x[cellIDs] + self.design.px[cellIDs]
    pin_y = y[cellIDs] + self.design.py[cellIDs]

    [Sxmax, dSxmax] = self.smooth_extremum(pin_x,  self.smoothing_param)
    [Symax, dSymax] = self.smooth_extremum(pin_y,  self.smoothing_param)
    [Sxmin, dSxmin] = self.smooth_extremum(pin_x, -self.smoothing_param)
    [Symin, dSymin] = self.smooth_extremum(pin_y, -self.smoothing_param)

    W   = (Sxmax - Sxmin) + (Symax - Symin)
    dWx = dSxmax - dSxmin
    dWy = dSymax - dSymin
    return (W, dWx, dWy)

  def callback(self, zk, state):
    dzk = zk - self.zk
    self.zk = zk
    

  def run(self):
    z = self.z0
    [f, df] = self.evaluate(z)
    for k in range(10):
      
      p = -df
      
      if any(p < 0):
        stepm = np.divide((z[p < 0] - self.lz[p < 0]), -p[p < 0]).min()
      else:
        stepm = self.design.R['dx']
      
      if any(p > 0):
        stepp = np.divide((self.uz[p > 0] - z[p > 0]),  p[p > 0]).min()
      else:
        stepp = self.design.R['dx']
      
      stepmax = min(stepm, stepp)

      step= stepmax
      while(1):
        znew = z + step*p
        [fnew, dfnew] = self.evaluate(znew)
        if fnew < f:
          break
        else:
          step *= 0.5

      z = znew
      f = fnew
      df = dfnew
      # self.callback(z, 0)
            
  def evaluate(self, z):
    x = z[self.xind]
    y = z[self.yind]


    [self.density, self.masks] = self.update_density(x, y, range(self.design.n_cells), self.density, self.masks)

    # relative_density = np.maximum(np.zeros_like(D), D - self.target_density*np.ones_like(D)) # Density relative to target
    
    [P, F, E] = self.compute_electrostatics(self.density.todense())

    # Determine the grid element(s) in which each cell locates, extract the
    # electric field at those locations, and scale by the cell charges (area). 
    force = {'x': np.zeros_like(x), 'y': np.zeros_like(y)}
    
    I = np.floor(x/self.dgx).astype(int)
    J = np.floor(y/self.dgy).astype(int)

    force['x'] = np.multiply(F['x'][I, J], self.charge)
    force['y'] = np.multiply(F['y'][I, J], self.charge)

    # self.plot_heatmap(P)
    # self.plot_cells(x, y)
    # plt.quiver(self.dgx*I, self.dgy*J, force['x'], force['y'])
    # for k in range(self.design.n_cells):
      # [I, J] = self.masks[k].nonzero()
      # force['x'][k] = np.multiply(F['x'][I, J], M[k][I, J]).sum()
      # force['y'][k] = np.multiply(F['y'][I, J], M[k][I, J]).sum()
      # force['x'][k] = F['x'][I, J].sum()
      # force['y'][k] = F['y'][I, J].sum()

    W    = 0.0
    dWx  = np.zeros_like(x)
    dWy  = np.zeros_like(y)

    for netID in range(self.design.n_nets):
      cellIDs = self.design.netlist[netID]
      [net_wawl, net_dWx, net_dWy] = self.net_WAWL(x, y, cellIDs)
      W += net_wawl
      dWx[cellIDs] += net_dWx
      dWy[cellIDs] += net_dWy
    
    dW =  np.concatenate((dWx, dWy))
    dE = -np.concatenate((force['x'], force['y'])) # Note that the force is the negative of the gradient

    # f  =  W + self.mu*E
    # df = dW + self.mu*E
    f = E
    df = dE

    self.trace['W'].append(W)
    self.trace['E'].append(E)
    self.trace['f'].append(f)
    self.trace['norm_dW'].append(np.linalg.norm(dW, ord=2))
    self.trace['norm_dE'].append(np.linalg.norm(dE, ord=2))
    self.trace['norm_df'].append(np.linalg.norm(df, ord=2))
    return (f, df)

  def solve(self):
    z0     = self.z0
    # z0     = self.sqzz
    lz     = np.zeros_like(z0)
    uz     = np.concatenate((self.design.R['dx'] - self.design.dx, self.design.R['dy'] - self.design.dy))

    result = minimize(fun=self.evaluate, x0=z0, method='Trust-constr', jac=True,  callback=self.callback, bounds=Bounds(lz, uz, keep_feasible=True), options={'disp': True, 'maxiter':100000})
    return result

  def update_density(self, x, y, cells, D, M):
    '''
    @brief: Update the density map by subtracting, shifting, and adding density masks of provided cells
    '''
    for k in cells:
      di = int(np.floor(x[k]/self.grid['dx'])) - M[k].nonzero()[0][0] # Change in row index
      dj = int(np.floor(y[k]/self.grid['dy'])) - M[k].nonzero()[1][0] # Change in column index

      if di != 0 or dj != 0: 
        D = D._add_sparse(-M[k]) # Subtract out this cells previous density contribution
        
        indices = M[k].indices + dj # Shift column indices 
          # M[k].indices += dj
    
        # Shift row index pointers (in CSR format)
        # For example,
        # shift   up 3 rows:  [0, 0, 0, 0, 4, 9, 13, 18, 18, 18, 18, 18] -> [0, 4, 9, 13, 18, 18, 18, 18, 18, 18, 18, 18]
        # shift down 2 rows:  [0, 0, 0, 0, 4, 9, 13, 18, 18, 18, 18, 18] -> [0, 0, 0,  0,  0,  0,  4,  9, 13, 18, 18, 18]
        if di > 0:
          pointers = np.concatenate((np.zeros(di), M[k].indptr[:-di])).astype(int)
          # M[k].indptr = np.concatenate((np.zeros(di), M[k].indptr[:-di])).astype(int)   
        elif di < 0: 
          # M[k].indptr = np.concatenate((M[k].indptr[abs(di):], M[k].nnz*np.ones(abs(di)))).astype(int)
          pointers = np.concatenate((M[k].indptr[abs(di):], M[k].nnz*np.ones(abs(di)))).astype(int)
        else: 
          pointers = M[k].indptr
        
        M[k] = scipy.sparse.csr_matrix((M[k].data, indices, pointers), shape=(self.grid['nx'], self.grid['ny']))

        D = D._add_sparse(M[k])
    return (D, M)

  def compute_density(self, x, y):
    D = scipy.sparse.csr_matrix((self.grid['nx'], self.grid['ny']))
    M = [None]*self.design.n_cells
    for k in range(self.design.n_cells):
      li = int(np.floor(x[k]/self.grid['dx']))
      lj = int(np.floor(y[k]/self.grid['dy']))
      [X, Y] = np.meshgrid(np.arange(li, li + self.dgx[k]), np.arange(lj, lj + self.dgy[k]), indexing='ij')
      M[k] = scipy.sparse.csr_matrix((np.ones(int(self.dgx[k]*self.dgy[k])).ravel(), (X.ravel(), Y.ravel())), shape=(self.grid['nx'], self.grid['ny']))
      D = D._add_sparse(M[k])
    return (D, M)

  # def compute_density(self, x, y):
  #   '''
  #   @brief   : computation of the cell density map based on the current placement
  #   @param x : current placement x-coordinates
  #   @param y : current placement y-coordinates
  #   '''
  #   D = scipy.sparse.csr_matrix((self.grid['nx'], self.grid['ny']))
  #   M = [None]*self.design.n_cells

  #   for k in range(self.design.n_cells):
  #     li = int(np.floor(x[k]/self.grid['dx']))
  #     lj = int(np.floor(y[k]/self.grid['dy']))
  #     ui = int(np.floor((x[k] + self.design.dx[k])/self.grid['dx']))
  #     uj = int(np.floor((y[k] + self.design.dy[k])/self.grid['dy']))
      
  #     di = ui - li
  #     dj = uj - lj      
  #     block = np.zeros((di + 1, dj + 1))      

  #     # Width and height of partial overlaps
  #     ldx = min((li + 1)*self.grid['dx'] - x[k], self.design.dx[k])          # lower horizontal
  #     ldy = min((lj + 1)*self.grid['dy'] - y[k], self.design.dy[k])          # lower vertical
  #     udx = (x[k] + self.design.dx[k]) - ui*self.grid['dx'] if di > 0 else 0 # upper horizontal
  #     udy = (y[k] + self.design.dy[k]) - uj*self.grid['dy'] if dj > 0 else 0 # upper vertical

  #     block[0,     0]   += ldx*ldy/self.grid['dxdy'] # lower left 
  #     block[0,    dj]   += ldx*udy/self.grid['dxdy'] # upper left 
  #     block[di,    0]   += udx*ldy/self.grid['dxdy'] # lower right
  #     block[di,   dj]   += udx*udy/self.grid['dxdy'] # upper right
  #     block[1:di,  0]   += ldy/self.grid['dy']       # lower horizontal strip
  #     block[0,  1:dj]   += ldx/self.grid['dx']       # left vertical strip
  #     block[di, 1:dj]   += udx/self.grid['dx']       # right vertical strip
  #     block[1:di, dj]   += udy/self.grid['dy']       # upper horizontal strip
  #     block[1:di, 1:dj] += 1.0                       # interior

  #     [X, Y] = np.meshgrid(np.arange(li, ui + 1), np.arange(lj, uj + 1), indexing='ij')
  #     M[k] = scipy.sparse.csr_matrix((block.ravel(), (X.ravel(), Y.ravel())), shape=(self.grid['nx'], self.grid['ny']))
  #     D = D._add_sparse(M[k])
  #   return (D, M)

  def compute_electrostatics(self, D):
    '''
    @brief : Computation of the system electrostatic quantities determined by the density map
    @param : D is a full (dctn doesn't work with sparse) matrix with the density value at each grid element 
    '''
    # Define the frequency mesh for the cosine bases
    u = (np.pi/self.grid['nx'])*np.arange(self.grid['nx'])
    v = (np.pi/self.grid['ny'])*np.arange(self.grid['ny'])
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
    F['x'] = idst(np.vstack((idct(Cu, axis=1, norm="ortho")[1:, :], np.zeros([1, self.grid['ny']]))), axis=0, norm="ortho")
     
    # Differentiating the potential map with respect to y gives the vertical
    # component of the electrostatic field. This is done by multiplying the
    # indices of C by elements of v.
    Cv = np.multiply(C, v)
    F['y'] = idct(idst(np.hstack((Cv[:, 1:], np.zeros([self.grid['nx'], 1]))), axis=1, norm="ortho"), axis=0, norm="ortho")

    E = 0.50*np.multiply(P, D).sum()    
    return (P, F, E)

  def plot_cells(self, x, y, cells="all"):
    if cells == "all":
      cells = range(self.design.n_cells)

    for i in cells:
      plt.gca().add_patch(Rectangle((x[i], y[i]), self.design.dx[i], self.design.dy[i], edgecolor='black', facecolor='none', lw=1.0, alpha=1.0) )

  def plot_heatmap(self, Z):
    X, Y = np.meshgrid(self.grid['dx']*np.arange(Z.shape[0]), self.grid['dy']*np.arange(Z.shape[1]), indexing='ij')
    hm = plt.gca().pcolormesh(X, Y, Z)
    plt.colorbar(hm)


  def plot_contour(self, Z):
    X, Y = np.meshgrid(self.grid['dx']*np.arange(Z.shape[0]), self.grid['dy']*np.arange(Z.shape[1]), indexing='ij')
    plt.gca().contour(X, Y, Z, 30)

  def plot_field(self, Z):
    X, Y = np.meshgrid(self.grid['dx']*np.arange(Z['x'].shape[0]), self.grid['dy']*np.arange(Z['y'].shape[1]), indexing='ij')
    plt.gca().quiver(X, Y, Z['x'], Z['y'])

  @staticmethod
  def plot_surface(Z):
    X, Y = np.meshgrid(np.arange(Z.shape[0]), np.arange(Z.shape[1]), indexing='ij')
    plt.gca(projection='3d').plot_surface(X, Y, Z)

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

    S_=z_norm*S
    dS_ = q + (S - np.dot(q, u))*u
    return (S_, dS_)

  @staticmethod
  def idsxt(X, axis=1):
    if axis == 0:
      X = X.T
    [m, n] = X.shape
    Y = np.zeros_like(X)
    for i in range(m):
      y = idct(np.flip(np.append(X[i, 1:], 0)), norm="ortho")
      y[1::2] = -y[1::2]
      Y[i,:] = y
    if axis == 0:
      Y = Y.T
    return Y

  @staticmethod
  def overlap(r1, r2):
    ''' 
    Computes the area of the intersection of two rectangles r1 and r2
    '''
    dx = max(0, min(r1['x'] + r1['dx'], r2['x'] + r2['dx']) - max(r1['x'], r2['x']))
    dy = max(0, min(r1['y'] + r1['dy'], r2['y'] + r2['dy']) - max(r1['y'], r2['y']))
    return dx*dy