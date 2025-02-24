#This code is licenced with MPL 2.0 
"""
Created on Mon Feb 24 2025

@author: jackm
"""

import numpy as np
import matplotlib.pyplot as plt

class MontiCarlo:
    
    def __init__(self, coords):
        "initialisation Constructor of n dimention array"
        self.coords = coords
    def __str__(self):
        "allows the coordinates to be printed"
        return str(self.coords)
    def __add__(self, other):
        "addition with other objects of same class"
        return MontiCarlo( np.add(self.coords , other.coords) )
    def __sub__(self, other):
        "subtraction of objects of same class"
        return MontiCarlo( np.subtract(self.coords, other.coords) )  
    def __mul__(self,other):
        "multiplication of objects of same class"
        return MontiCarlo(self.coords*other.coords)
    def __pow__(self, power):
        "takes self to the power of any number"
        return MontiCarlo( self.coords**power )
    def __getitem__(self, index):
        "alllows the seperation of the coordinates"
        return self.coords[index]
    
    def point_radius(self):
        "the radius position of the coordinate from the centre"
        rad_squared = np.sum(self.coords**2)
        return np.sqrt(rad_squared)
    
    def sq_boundary(self):
        "the square boundary conditions"
        lengths = 2*self.point_radius()
        area = lengths**2
        return area
    # def in_circle(self, sq_boundary_lenth):
    #     "checks if the point is inside the circle of boundary"
    #     if self.point_radius() <= boundary_length.point_radius():
            

arr_2d = np.array([ 4, 4 ])
test_2d = MontiCarlo(arr_2d)

test_2d1 = MontiCarlo(arr_2d)
a = test_2d**2  - test_2d1
b = test_2d.point_radius()
print(b)


t3 = test_2d.sq_boundary()





