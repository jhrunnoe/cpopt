from Rect import Rect

class Cell(Rect):
    def __init__(self, placer, ID):
      super().__init__(placer.x[ID], placer.y[ID], placer.dx[ID], placer.dy[ID])
      self.placer = placer
      self.ID = ID

    @property
    def posx(self):
      return self.placer.x[self.ID]
    
    @property
    def posy(self):
      return self.placer.y[self.ID]
    
    @property
    def pinx(self):
      return self.posx + 0.50*self.placer.dx[self.ID]
    
    @property
    def piny(self):
      return self.posy + 0.50*self.placer.dy[self.ID]