##
# @file     Design.py
# @author   Jeb Runnoe
# @date     Feb 2023
# @brief    Design class for organizing, extracting, loading, and saving design data
#           relevant to cell placement optimization
#

import numpy as np
import gzip
import json
import pickle
import os
from scipy.sparse import coo_matrix

class Design:
  ''' 
  @brief A Design class for organizing, extracting, loading, and saving chip design information relevant to cell placement optimization. 
      - R       : The placement region 
      - netlist : Netlist that defines instance connectivity
      - n_cells : The number of cells (instances)
      - n_nets  : The number of nets in the netlist
      - (x0, y0): Initial (x, y)-coordinates of the center of all instances
      - (dx, dy): Physical dimensions of all instances
      - (px, py): Pin coordinates as offsets relative to the anchor point (can be negative)
      Each cell has the following geometry:
       (x-dy/2, y+dy/2)+------------+-----------+(x+dx/2, y+dy/2)
                      |             |           |
                      |             |           |
                      +-----------(x,y)--+------+
                      |             |    |      |
                      |             |    |      |
       (x-dy/2,y-dy/2)+-------------+----@------+(x+dx/2, y-dy/2)
                                    (x+px,y+py)
      '''
  def __init__(self, name):
    self.name = name
    self.path = os.path.dirname(os.path.abspath(__file__))
    self.design_path = self.path + '/NCSU-DigIC-GraphData-2022-10-15/'
    # self.design_path = self.path +'/RosettaStone-GraphData-2023-05-23/'
 
    self.dir  = self.design_path + self.name + '/'
    self.file_name = self.name + '_design_data'
    

  def extract(self, file_name=None):
    if file_name is None:
      file_name = self.file_name

    if os.path.isfile(file_name + '.pkl'):
      # The optimization data has already been extracted - load it from the saved copy
      with open(file_name + '.pkl', 'rb') as fid:
        data = pickle.load(fid)
        self.R          = data.R
        self.x0         = data.x0
        self.y0         = data.y0
        self.dx         = data.dx
        self.dy         = data.dy
        self.px         = data.px
        self.py         = data.py
        self.netlist    = data.netlist
        self.n_cells    = data.n_cells
        self.n_nets     = data.n_nets
        self.cell_names = data.cell_names
        self.net_names  = data.net_names
    else:
      # Instance dimensions and initial placement extraction
      with gzip.open(self.design_path + 'cells.json.gz', 'rb') as fid:
        cells = json.loads(fid.read().decode('utf-8'))

      with gzip.open(self.dir + self.name + '.json.gz', 'rb') as fid:
        spec = json.loads(fid.read().decode('utf-8'))

      self.n_cells = len(spec['instances'])
      self.n_nets  = len(spec['nets'])

      self.x0 = np.empty(self.n_cells)
      self.y0 = np.empty(self.n_cells)
      self.dx = np.empty(self.n_cells)
      self.dy = np.empty(self.n_cells)
      self.px = np.empty(self.n_cells)
      self.py = np.empty(self.n_cells)

      self.cell_names = [None]*self.n_cells
  
      for instance in spec['instances']:
        xloc   = instance['xloc']
        yloc   = instance['yloc']
        index  = instance['cell'] # For looking up dimensions of given cell type
        orient = instance['orient']
        i      = instance['id'] # The cell id
        
        width  = cells[index]['width']
        height = cells[index]['height']
        
        (dx, dy) = (width, height)

        terminals  = cells[index]['terms']
        if 'xloc' in terminals and 'yloc' in terminals:
          px = np.mean([term['xloc'] for term in terminals])
          py = np.mean([term['yloc'] for term in terminals])
        else: # If terminals not specified by design, use the cell midpoint
          px = 0.50*dx
          py = 0.50*dy

        if orient in [1, 3, 5, 7]:
          (dx, dy) = (height, width)
          (px, py) = (py, px)

        if orient in [1, 2, 4, 5]:
          xloc -= dx # lower x coordinate after rotation/reflection
        if orient in [2, 3, 5, 6]:
          yloc -= dy # lower y coordinate after rotation/reflection

        self.x0[i] = xloc + 0.50*dx
        self.y0[i] = yloc + 0.50*dy
        self.dx[i] = dx
        self.dy[i] = dy
        self.px[i] = px - 0.50*dx
        self.py[i] = py - 0.50*dy
        self.cell_names[i] = instance['name'] 

      self.R = dict.fromkeys(['x', 'y', 'dx', 'dy'])
      self.R['dx'] = (1.10*(self.x0 + 0.50*self.dx).max()).round()
      self.R['dy'] = (1.10*(self.y0 + 0.50*self.dy).max()).round()
      self.R['x']  = 0.50*self.R['dx']
      self.R['y']  = 0.50*self.R['dy']

      # Netlist extraction
      conn = np.load(self.dir + self.name + '_connectivity.npz')
      coo  = coo_matrix((conn['data'], (conn['row'], conn['col'])), shape = conn['shape'])
      
      self.net_names = [None]*self.n_nets
      self.netlist   = [None]*self.n_nets
      for j in range(self.n_nets):
        self.net_names[j] = spec['nets'][j]['name']
        self.netlist[j] = coo.row[np.where(coo.col == j)]
    
      self.save(file_name)

  def save(self, file_name=None):
    if file_name is None:
      file_name = self.file_name
    with open(file_name + '.pkl', 'wb') as fid: 
      pickle.dump(self, fid)