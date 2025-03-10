#This code is licenced with MPL 2.0
"""
Created on Mon Feb 24 2025

@author: jackm
"""

import numpy as np
import matplotlib.pyplot as plt
from math import fsum
 
class MontiCarlo:

    def __init__(self, coords, boundary):
        
        """initialisation Constructor of n dimention array
        Parameters
        ----------
        coords : list of co-ordinates in n dimentions , shape (n,num_points).
        boundary : list of boundary conditions for the integral"""
        
        self.coords = coords
        self.boundary = boundary
        
    def __str__(self):
        "allows the coordinates to be printed"
        return str(self.coords)
    def __add__(self, other):
        "addition with other objects of same class"
        return MontiCarlo( np.add(self.coords , other.coords), self.boundary )
    def __sub__(self, other):
        "subtraction of objects of same class"
        return MontiCarlo( np.subtract(self.coords, other.coords), self.boundary )
    def __mul__(self,other):
        "multiplication of objects of same class"
        return MontiCarlo(self.coords*other.coords,self.boundary)
    def __pow__(self, power):
        "takes self to the power of any number"
        return MontiCarlo( (self.coords**power),self.boundary )
    def __getitem__(self, index):
        "alllows the seperation of the coordinates"
        return self.coords[index]    

    def used_points(self):

        self.inclosed_points = np.delete(self.coords, np.where(self.coords[0]**2+self.coords[1]**2
                                        > self.radius**2),axis = 1)
        self.out_points = np.delete(self.coords, np.where(self.coords[0]**2+self.coords[1]**2
                                        < self.radius**2),axis = 1)

        self.ratio = fsum( (np.where(self.coords[0]**2+self.coords[1]**2 < self.radius**2
                              , 1, 0) )/ (len(self.coords[0])) )
        return

    def integrate(self, func, num):
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

        self.integral = (self.boundary[1]-self.boundary[0])*np.mean(func(self.coords))
        return self.integral
    
    def var_std(self):
        self.var = np.var(self.coords)
        self.std = np.sqrt(self.var)
        # self.var = (self.boundary[1]-self.boundary[0])**2/12
        # self.std = np.sqrt(self.var)
        return self.var,self.std

    def point_radius(self):
        "the radius position of the coordinate from the centre"
        self.radius = (abs(self.boundary[0])+abs(self.boundary[1]))/2
        return self.radius
    
    def sq_boundary(self,boundary):
        "the square boundary conditions"
        lengths = 2*self.point_radius()
        area = lengths**2
        return area

    def plot1d(self,func):
        

        x_points = np.linspace(self.boundary[0], self.boundary[1],100)
        f_est =  np.empty(np.shape(x_points))
        
        for i in range(len(x_points)):
            samples = np.random.uniform(self.boundary[0], x_points[i], 1000) 

            f_est[i]= (x_points[i]-self.boundary[0])*np.mean(func(samples))
        
        plt.figure()
        plt.plot(x_points,f_est ,'o')
        plt.xlabel('x points')
        plt.ylabel('Anti Derivitive of F(x)')
        return
    
    def plotcirc(self):
        "Plots the special case of a circle"
        self.used_points()
        x_square,y_square = self.radius,self.radius
        square = [ [-x_square, -x_square, x_square, x_square, -x_square]
                  ,[-y_square, y_square , y_square, -y_square, -y_square] ]


        theta = np.linspace(0, 2*np.pi,100)
        x_circ = self.radius*np.cos(theta) 
        y_circ = self.radius*np.sin(theta) 
        
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


if __name__ == "__main__":
    rad = 1
    low_lim = -rad
    up_lim  = rad
    N = 1000000
    x_arr =  np.random.uniform(low_lim, up_lim , size =N)
    y_arr = np.random.uniform(low_lim, up_lim , size=N)
    bounds = np.array([low_lim,up_lim])
    
    print('\n1D testing')
    arr_1d = np.array([x_arr])
    test_1d = MontiCarlo(arr_1d, bounds)
    I =test_1d.integrate(sin, N)
    test_1d.plot1d(sin)
    print(f'integral check = {I} \nvar & std = {test_1d.var_std()}')    

    print('\n2D testing')
    arr_2d = np.array([x_arr, y_arr])
    test_2d = MontiCarlo(arr_2d, bounds)

    test_2d1 = MontiCarlo(arr_2d, bounds)
    a = test_2d  - test_2d1
    b = test_2d.point_radius()    
    test_2d.plotcirc()
    print(f'integral check = {test_2d.integrate(circ,N)}')
    print(f'ratio = {test_2d.ratio}, pi = {test_2d.ratio*4}')
    print(f'var & std = {test_2d.var_std()}')



    



    # def in_circle(self, sq_boundary_lenth):
    #     "checks if the point is inside the circle of boundary"
    #     if self.point_radius() <= boundary_length.point_radius():
      