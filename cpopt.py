import core

def main():
  plcr = core.Placer('counter')
  plcr.Density(plcr.x, plcr.y)
  plcr.densityHeatMap()

if __name__ == "__main__":
  main()