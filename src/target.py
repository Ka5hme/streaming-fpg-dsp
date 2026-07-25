import numpy as np
import matplotlib.pyplot as plt

class Target:
    def __init__(self, range_m:float, rcs:float, velocity:float=0):
        self.range_m = range_m
        self.rcs = rcs
        self.velocity = velocity

        if(range_m<0):
            raise ValueError("Range must be greater than 0")

        if(rcs<0):
            raise ValueError("RCS must be greater than 0")