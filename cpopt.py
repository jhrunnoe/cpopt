from Placer import Placer
import numpy as np
from Design import Design
import time
import matplotlib.pyplot as plt

def main():
  placer = Placer('counter')
  placer.evaluate(placer.design.z0)
  
if __name__ == "__main__":
  main()