#!/usr/bin/env python3
#This code is licenced with MPL 2.0
"""
Created on Mon Feb 24 2025

@author: jackm
"""

import numpy as np
import matplotlib.pyplot as plt



class MonteCarlo:

    def __init__(self, coords, boundary):

        """    
        Monte Carlo class which calculates integrals of functionsa and 
        coordinates in n dimentions.
        
        initialisation Constructor Parameters
        ----------
        coords : list of co-ordinates in n dimensions , shape (n,num_points).
        boundary : list of boundary conditions for the integral
        """

        self.coords = coords
        self.boundary = boundary
        self.dim = len(coords[:,0])

    def __str__(self):
        "allows the coordinates to be printed"
        return str(self.coords)

    def __getitem__(self, index):
        "allows the seperation of the coordinates into each dimension"
        return self.coords[index]


    def integrate(self, func):
        """
        Parameters
        ----------
        func : Arbirary funcion which passes the coords.
        boundary : List of the integral limits (i.e b and a).

        Returns
        -------
        Value of integrated random points.
        """
        integral = (self.boundary[1]-self.boundary[0])**self.dim *np.mean(func(self.coords))
        return integral

    def mean_var_std(self,func):
        "calculates the mean, varience and standard deviation"
        self.f_array = func(self.coords)
        self.mean = np.mean(self.f_array)
        var = np.var(self.f_array)
        std = np.sqrt(var) * (self.boundary[1]-self.boundary[0])**self.dim
        return self.mean, var,std


    def plotcirc(self):
        "Plots the special case of the dart board example"
        radius = (abs(self.boundary[0])+abs(self.boundary[1]))/2

        inclosed_points = np.delete(self.coords, np.where(self.coords[0]**2+
                                    self.coords[1]**2 > radius**2),axis = 1)

        out_points = np.delete(self.coords, np.where(self.coords[0]**2+
                                    self.coords[1]**2 < radius**2),axis = 1)

        x_square,y_square = radius,radius
        square = [ [-x_square, -x_square, x_square, x_square, -x_square]
                  ,[-y_square, y_square , y_square, -y_square, -y_square] ]


        theta = np.linspace(0, 2*np.pi,100)
        x_circ = radius*np.cos(theta)
        y_circ = radius*np.sin(theta)

        plt.figure()
        plt.plot(inclosed_points[0], inclosed_points[1], 'rx')
        plt.plot(square[0],square[1],'k')
        plt.plot(out_points[0], out_points[1], 'bx')
        plt.plot(x_circ,y_circ,'k')
        plt.axis('square')
        plt.show()

    def plot1d(self,func):
        "Plots the anti Derivitive of a 1D function (eg. sin(x) dx = -cos(x))"
        x_points = np.linspace(self.boundary[0], self.boundary[1],100)
        f_est =  np.empty(np.shape(x_points))

        for i in range(len(x_points)):
            samples = np.random.uniform(self.boundary[0], x_points[i], 1000)
            f_est[i]= (x_points[i]-self.boundary[0])*np.mean(func(samples))

        plt.figure()
        plt.plot(x_points,f_est ,'o')
        plt.xlabel('x points')
        plt.title('Anti Derivitive of F(x)')
        plt.show()



if __name__ == "__main__":

    ###########################################################################
    #Initial Conditions
    ###########################################################################
    RAD = 1
    LOW_LIM = -RAD
    UP_LIM  = RAD
    N = 10000
    x_arr =  np.random.uniform(LOW_LIM, UP_LIM , size =N)
    y_arr = np.random.uniform(LOW_LIM, UP_LIM , size=N)

    LOW_LIM = -1
    UP_LIM  = 1
    bounds = np.array([LOW_LIM,UP_LIM])

    def sin(x):
        "A simple sin function for testing"
        return np.sin(x)

    def circ(coords):
        "circle function for estimating pi/4"
        rad_point = np.sqrt( np.sum(coords**2, axis = 0) )
        radius = 1
        rad_arr = np.where(rad_point < radius,1,0)
        return rad_arr

    ###########################################################################
    #Testing for non parallel computations
    ###########################################################################
    print('\n1D testing')
    arr_1d = np.array([x_arr])
    test_1d = MonteCarlo(arr_1d, bounds)
    Int =test_1d.integrate(sin)
    test_1d.plot1d(sin)
    print(f'integral check = {Int} \nmean, var & std = {test_1d.mean_var_std(sin)}')

    print('\n2D testing')
    arr_2d = np.array([x_arr, y_arr])
    test_2d = MonteCarlo(arr_2d, bounds)

    test_2d1 = MonteCarlo(arr_2d, bounds)
    test_2d.plotcirc()
    print(f'integral check = {test_2d.integrate(circ)}')
    print(f'mean,var & std = {test_2d.mean_var_std(circ)}')
