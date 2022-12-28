import numpy as np
from scipy.sparse import coo_matrix
import json
import gzip
import re

class Cell:
  def __init__(self, x, y, dx, dy, ID = None):
    self.ID = ID # Interger between 0 and nC - 1 where nC = |C| is the number of cells
    self.x  = x
    self.y  = y
    self.dx = dx
    self.dy = dy

    # Temporarily using the center for the pin location
    self.px = x + 0.50*dx
    self.py = y + 0.50*dy


class Net:
  def __init__(self, cells):
    self.cells = cells
    self.size = len(self.cells)

    self.IDs = [None]*self.size
    self.x = np.empty((self.size, 1))
    self.y = np.empty((self.size, 1))
    for i in range(self.size):
      self.IDs[i] = cells[i].ID
      self.x[i] = cells[i].x
      self.y[i] = cells[i].y
      
  def HPWL(self):
    dx = self.x.max() - self.x.min()
    dy = self.y.max() - self.y.min()
    return 0.50*(dx + dy)


class Design:
  '''The Design class specifies the chip design information 
     relevant to cell placement optimization'''
  def __init__(self, designName):
    self.name = designName
    self.path = './NCSU-DigIC-GraphData-2022-10-15/'
    self.R = self.SetupRegion()
    self.C = self.SetupCells()
    self.N = self.SetupNetList()

  def SetupRegion(self):
    with open(self.path + self.name + '/' + self.name + '_route_opt.def') as f:
      for line in f:
        if 'DIEAREA' in line:
          die = [int(s) for s in re.findall(r'\d+', line)] # This gets numeric values from the die area line
          Rx  = die[0]
          Ry  = die[1]
          dRx = die[4] - Rx
          dRy = die[5] - Ry
          return Cell(Rx, Ry, dRx, dRy)
  
  def SetupCells(self):
    with gzip.open(self.path + 'cells.json.gz', 'rb') as f:
      cells = json.loads(f.read().decode('utf-8'))

    with gzip.open(self.path + self.name + '/' + self.name + '.json.gz', 'rb') as f:
      design = json.loads(f.read().decode('utf-8'))

    # Extract instance coordinates & dimensions
    C  = list()

    for instance in design['instances']:
      xloc   = instance['xloc']
      yloc   = instance['yloc']
      index  = instance['cell'] # For looking up dimensions of this cell type
      orient = instance['orient']

      width  = cells[index]['width']
      height = cells[index]['height']
      i = instance['id'] # The cell id

      (dx, dy) = (height, width) if orient in [1, 3, 5, 7] else (width, height)

      if orient in [1, 2, 4, 5]:
        xloc -= dx # lower x coordinate after rotation/reflection
      if orient in [2, 3, 5, 6]:
        yloc -= dy # lower y coordinate after rotation/reflection

      C.append(Cell(xloc, yloc, dx, dy, i))
    return C

  def SetupNetList(self):
    conn = np.load(self.path + self.name + '/' + self.name + '_connectivity.npz')
    coo = coo_matrix((conn['data'], (conn['row'], conn['col'])), shape = conn['shape'])
    self.nC, self.nN = coo.get_shape()
    N = list()
    for j in range(self.nN):
      IDs = coo.row[np.where(coo.col == j)]
      N.append(Net([self.C[i] for i in IDs]))
    return N

  def HPWL(self):
    return sum(n.HPWL() for n in self.N)
