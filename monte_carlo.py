#!/usr/bin/env python3
#This code is licenced with MPL 2.0
"""
Created on Mon Feb 24 2025

@author: jackm
"""

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI


class MonteCarlo:
    """
    Monte Carlo class which calculates integrals of functionsa and coordinates
    in n dimentions .
    """

    def __init__(self, coords, boundary):

        """initialisation Constructor of n dimention array
        Parameters
        ----------
        coords : list of co-ordinates in n dimensions , shape (n,num_points).
        boundary : list of boundary conditions for the integral
        dim: n dimensions
        """

        self.coords = coords
        self.boundary = boundary
        self.dim = len(coords[:,0])

    def __str__(self):
        "allows the coordinates to be printed"
        return str(self.coords)

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



class ParallelMonteCarlo(MonteCarlo):
    """
    A sub class of MonteCarlo enabling parallel opperations using MPI
    """

    def __init__(self, n_per_rank,boundaries, dimensions = int):

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()


        self.procs = self.comm.Get_size()

        self.total_points = n_per_rank*self.procs


        self.points_per_rank  = np.random.uniform(boundaries[0],
                                                      boundaries[1],
                                                      n_per_rank)

        n_coords_per_rank = len(self.points_per_rank) // dimensions
        coords_per_rank = self.points_per_rank[:n_coords_per_rank * dimensions]
        coords_per_rank = coords_per_rank.reshape(dimensions, n_coords_per_rank)


        super().__init__(coords_per_rank,boundaries)


    def parallel_integrate(self, func):
        "enables each rank to integral with the mean,varience and error(std)"
        local_integral = self.integrate(func)

        local_stats = self.mean_var_std(func)

        n_total = len(self.coords)*len(self.coords[0])

        par_integral = self.comm.reduce(local_integral, op = MPI.SUM , root = 0 )

        expected_val = local_stats[0]
        expected_val_squared = np.mean(self.f_array**2)

        par_expected_val = self.comm.reduce(expected_val,
                                                 op = MPI.SUM, root = 0)

        par_expected_val_squared = self.comm.reduce(expected_val_squared,
                                                 op = MPI.SUM, root = 0)

        if self.rank == 0:

            par_integral = par_integral /self.procs

            boundary_dim = (self.boundary[1] - self.boundary[0])**self.dim

            var = 1/n_total *( (par_expected_val_squared/self.procs)
                                   - (par_expected_val/self.procs)**2 )

            error = np.sqrt(var) * boundary_dim

            print(f'\n{self.dim} dimentional {func.__name__}',
                  '\n-------------------------',
                  f'\nIntegral = {par_integral}',
                  f'\nMean = {expected_val}',
                  f'\nVar = {var}',
                  f'\nStd = {error}')

            return par_integral, expected_val, var, error

        return None

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
