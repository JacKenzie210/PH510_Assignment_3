#This code is licenced with MPL 2.0
"""
Created on Mon Feb 24 2025

@author: jackm
"""

import numpy as np
import matplotlib.pyplot as plt
from math import fsum
 
class MontiCarlo:

    def __init__(self, coords):
        
        """initialisation Constructor of n dimention array
        Parameters
        ----------
        coords : co-ordinates in n dimentions , shape (n,num_points)."""
        self.coords = coords
        
    def used_points(self, radius):
        
        self.inclosed_points = np.delete(self.coords, np.where(self.coords[0]**2+self.coords[1]**2
                                        > radius**2),axis = 1)
        self.out_points = np.delete(self.coords, np.where(self.coords[0]**2+self.coords[1]**2
                                        < radius**2),axis = 1)
        
        self.ratio = fsum( (np.where(self.coords[0]**2+self.coords[1]**2 < radius**2
                              , 1, 0) )/ (len(self.coords[0])) )
        
        return

    def integrate(self, func, boundary, num):
        """
        Parameters
        ----------
        func : Arbirary funcion which passes the coords.
        boundary : List of the integral limits (i.e b and a).
        num : number of points.

        Returns
        -------
        Value of integrated random points.
        """

        # self.inclosed_points = np.delete(self.coords, np.where(boundary[0] < func(self.coords) < boundary[1]),axis = 1)
        # self.out_points = np.delete(self.coords, np.where(self.coords[0]**2+self.coords[1]**2
        #                                 < radius**2),axis = 1)
        
        # self.ratio = fsum( (np.where(self.coords[0]**2+self.coords[1]**2 < radius**2
        #                       , 1, 0) )/ (len(self.coords[0])) )     

        
        result = (boundary[1]-boundary[0])*np.mean(func(self.coords))
        
        
        return result
        
    
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
        rad_squared = np.sum(self.coords**2, axis = 0)
        return np.sqrt(rad_squared)
    
    def sq_boundary(self,radius):
        "the square boundary conditions"
        lengths = 2*self.point_radius()
        area = lengths**2
        return area

    def plot1d(self,func, boundary):
        

        x_points = np.linspace(boundary[0], boundary[1],100)
        f_est =  np.empty(np.shape(x_points))
        
        for i in range(len(x_points)):
            samples = np.random.uniform(boundary[0], x_points[i], 1000) 

            f_est[i]= (x_points[i]-boundary[0])*np.mean(func(samples))
        
        plt.figure()
        plt.plot(x_points,f_est ,'o')
        return
    
    def plotcirc(self,radius):
        "Plots the function if it is 2D"
        points = self.used_points(radius)

        x_square,y_square = radius,radius
        square = [ [-x_square, -x_square, x_square, x_square, -x_square]
                  ,[-y_square, y_square , y_square, -y_square, -y_square] ]


        theta = np.linspace(0, 2*np.pi,100)
        x_circ = radius*np.cos(theta)
        y_circ = radius*np.sin(theta)
        
        plt.figure()
        plt.plot(self.inclosed_points[0], self.inclosed_points[1], 'rx')
        plt.plot(square[0],square[1],'k')
        plt.plot(self.out_points[0], self.out_points[1], 'bx')
        plt.plot(x_circ,y_circ,'k')
        plt.axis('square')

        
        
        

def sin(x):
    "simple sin function for test"
    return np.sin(x)
def circ(coords):
    "circle function for estimating pi/4"
    return coords[0]**2+ coords[1]**2


rad = np.pi
low_lim = 0
up_lim  = 2*rad
N = 10000
x_arr =  np.random.uniform(low_lim, up_lim , size =N)
y_arr = np.random.uniform(low_lim, up_lim , size=N)

arr_2d = np.array([x_arr, y_arr])
test_2d = MontiCarlo(arr_2d)

test_2d1 = MontiCarlo(arr_2d)
a = test_2d  - test_2d1
print(a)
b = test_2d.point_radius()
print(b)

t3 = test_2d.sq_boundary(rad)

#test_2d.plot1d(sin,[low_lim,up_lim])

#print(f'ratio = {test_2d.ratio}')

arr_1d = np.array([x_arr])
test_1d = MontiCarlo(arr_1d)
I =test_1d.integrate(sin, [low_lim,up_lim], N)
test_1d.plot1d(sin,[low_lim,up_lim])
print(f'integral check = {I}')



    
    
    
    
    
    # def in_circle(self, sq_boundary_lenth):
    #     "checks if the point is inside the circle of boundary"
    #     if self.point_radius() <= boundary_length.point_radius():
            