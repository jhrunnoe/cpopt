from Placer import Placer

def main():
  placer = Placer('xbar')
  placer.ComputeDensity() # Compute the current density map
  placer.ComputeES()
  placer.PlotHeatmap()
  placer.PlotContour()
  placer.PlotSurface()

  1
  
if __name__ == "__main__":
  main()