#This code is licenced with MPL 2.0
"""
Created on Mon Feb 24 2025

@author: jackm
"""

from math import fsum
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
 
class MontiCarlo:

    def __init__(self, coords, boundary):
        
        """initialisation Constructor of n dimention array
        Parameters
        ----------
        coords : list of co-ordinates in n dimentions , shape (n,num_points).
        boundary : list of boundary conditions for the integral
        dim: n dimentions
        """

        self.coords = coords
        self.boundary = boundary
        self.dim = len(coords[:,0])
        self.N = len(coords)*len(coords[0])

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

        self.integral = (self.boundary[1]-self.boundary[0])**self.dim *np.mean(func(self.coords))
        return self.integral

    def mean_var_std(self,func):
        "calculates the mean, varience and standard deviation"
        self.f_array = func(self.coords)
        self.mean = np.mean(self.f_array)
        self.var = np.var(self.f_array)
        self.std = np.sqrt(self.var) * (self.boundary[1]-self.boundary[0])**self.dim
        return self.mean, self.var,self.std

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
        plt.title(f'Anti Derivitive of F(x)')
        return

    def circ_points(self):

        self.inclosed_points = np.delete(self.coords, np.where(self.coords[0]**2+self.coords[1]**2
                                        > self.radius**2),axis = 1)
        self.out_points = np.delete(self.coords, np.where(self.coords[0]**2+self.coords[1]**2
                                        < self.radius**2),axis = 1)

        self.ratio =  fsum( (np.where(self.coords[0]**2 +self.coords[1]**2< self.radius**2
                              , 1, 0) )/ (len(self.coords[0]))  )
        return 

    def plotcirc(self):
        "Plots the special case of a circle"
        self.circ_points()
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

#A subclass to use MPI example
class ParallelMontiCarlo:

    def __init__(self, n_per_rank,boundaries):
        
        comm = MPI.COMM_WORLD
        self.ranks = comm.Get_rank()
        self.procs = comm.Get_size()
        print(self.ranks,self.procs)

        self.n_per_rank = n_per_rank
        self.boundaries = boundaries
        self.coords_per_rank  = np.random.uniform(self.boundaries[0],self.boundaries[1],
                                                  self.N_per_rank)
        MontiCarlo.__init__(self,self.coords_per_rank,self.boundaries)




def sin(xvals):
    "simple sin function for test"
    return np.sin(xvals)
def circ(coords):
    "circle function for estimating pi/4"
    rad_point = np.sqrt( np.sum(coords**2, axis = 0) )
    radius = rad
    rad_arr = np.where(rad_point < radius,1,0)
    ratio = fsum(rad_arr)/len(rad_arr)
    return rad_arr


if __name__ == "__main__":
    rad = 1
    low_lim = -rad
    up_lim  = rad
    N = 10000
    x_arr =  np.random.uniform(low_lim, up_lim , size =N)
    y_arr = np.random.uniform(low_lim, up_lim , size=N)
    bounds = np.array([low_lim,up_lim])

    print('\n1D testing')
    arr_1d = np.array([x_arr])
    test_1d = MontiCarlo(arr_1d, bounds)
    I =test_1d.integrate(sin)
    test_1d.plot1d(sin)
    print(f'integral check = {I} \nmean, var & std = {test_1d.mean_var_std(sin)}')

    print('\n2D testing')
    arr_2d = np.array([x_arr, y_arr])
    test_2d = MontiCarlo(arr_2d, bounds)

    test_2d1 = MontiCarlo(arr_2d, bounds)
    a = test_2d  - test_2d1
    b = test_2d.point_radius()
    test_2d.plotcirc()
    print(f'integral check = {test_2d.integrate(circ)}')
    print(f'ratio = {test_2d.ratio}, pi = {test_2d.ratio*4}')
    print(f'mean,var & std = {test_2d.mean_var_std(circ)}')
    
    
    
    print('\n2D parallel Testing')
    num_per_rank = 1000
    test_par = (num_per_rank, bounds)
    print(test_par)
    
    
    