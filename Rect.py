class Rect:
    def __init__(self, x, y, dx, dy):
      # Location of bottom left corner
      self.x = x
      self.y = y
      # Width, height, area
      self.dx   = dx
      self.dy   = dy
      self.area = dx*dy
      # Location of center
      self.cx = x + 0.50*dx
      self.cy = y + 0.50*dy