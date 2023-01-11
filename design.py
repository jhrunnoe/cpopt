import numpy as np
import re
import gzip
import json
from scipy.sparse import coo_matrix

class Design:
  ''' 
  The Design class specifies the chip design information relevant to cell placement optimization. 
      - R       : The placement region 
      - (x0, y0): Initial (x, y)-coordinates of all instances
      - (dx, dy): Physical dimensions of all instances
      - N       : Netlist that defines instance connectivity
      - nCells : The number of cells (instances)
      - nNets  : The number of nets in the netlist
  '''
  def __init__(self, name):
    self.name = name
    self.path = './NCSU-DigIC-GraphData-2022-10-15/'
    self.dir  = self.path + self.name + '/' + self.name

    # Read the placement region coordinates from the .def file
    with open(self.dir + '_route_opt.def') as f:
      for line in f:
        if 'DIEAREA' in line:
          die = [int(s) for s in re.findall(r'\d+', line)] # This gets numeric values from the die area line
          self.R = {'x': die[0], 'y': die[1], 'dx': die[4] - die[0], 'dy': die[5] - die[1]}
          break
    
    # Instance dimensions and initial placement extraction
    with gzip.open(self.path + 'cells.json.gz', 'rb') as f:
      cells = json.loads(f.read().decode('utf-8'))

    with gzip.open(self.dir + '.json.gz', 'rb') as f:
      spec = json.loads(f.read().decode('utf-8'))

    self.nCells = len(spec['instances'])
    self.nNets  = len(spec['nets'])

    self.x0 = np.empty(self.nCells)
    self.y0 = np.empty(self.nCells)
    self.dx = np.empty(self.nCells)
    self.dy = np.empty(self.nCells)

    self.cellNames = [None]*self.nCells

    for instance in spec['instances']:
      xloc   = instance['xloc']
      yloc   = instance['yloc']
      index  = instance['cell'] # For looking up dimensions of given cell type
      orient = instance['orient']

      width  = cells[index]['width']
      height = cells[index]['height']
      i = instance['id'] # The cell id

      (dx, dy) = (height, width) if orient in [1, 3, 5, 7] else (width, height)

      if orient in [1, 2, 4, 5]:
        xloc -= dx # lower x coordinate after rotation/reflection
      if orient in [2, 3, 5, 6]:
        yloc -= dy # lower y coordinate after rotation/reflection

      self.x0[i] = xloc
      self.y0[i] = yloc
      self.dx[i] = dx
      self.dy[i] = dy
      self.cellNames[i] = instance['name'] 

    # Netlist extraction
    conn = np.load(self.dir + '_connectivity.npz')
    coo  = coo_matrix((conn['data'], (conn['row'], conn['col'])), shape = conn['shape'])
    
    self.netNames = [None]*self.nNets
    self.N        = [None]*self.nNets
    for j in range(self.nNets):
      self.netNames[j] = spec['nets'][j]['name']
      self.N[j] = coo.row[np.where(coo.col == j)]