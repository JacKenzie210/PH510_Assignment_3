#!/usr/bin/env python3
#This code is licenced with MPL 2.0
"""
Created on Mon Feb 24 2025

@author: jackm
"""

from math import fsum
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


    def circ_points(self):
        "Used to calculate the ratio of points inside and outside the circle"
        self.radius = (abs(self.boundary[0])+abs(self.boundary[1]))/2

        self.inclosed_points = np.delete(self.coords, np.where(self.coords[0]**2+self.coords[1]**2
                                        > self.radius**2),axis = 1)
        self.out_points = np.delete(self.coords, np.where(self.coords[0]**2+self.coords[1]**2
                                        < self.radius**2),axis = 1)

        self.ratio =  fsum( (np.where(self.coords[0]**2 +self.coords[1]**2< self.radius**2
                              , 1, 0) )/ (len(self.coords[0]))  )

    def plotcirc(self):
        "Plots the special case of the dart board example"
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



class ParallelMonteCarlo(MonteCarlo):
    """
    A sub class of MonteCarlo enabling parallel opperations using MPI
    """

    def __init__(self, n_per_rank,boundaries, dimensions = int):

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()


        self.procs = self.comm.Get_size()
        self.n_per_rank = n_per_rank
        self.total_points = self.n_per_rank*self.procs

        self.boundaries = boundaries
        self.points_per_rank  = np.random.uniform(self.boundaries[0],
                                                      self.boundaries[1],
                                                      self.n_per_rank)

        n_coords_per_rank = len(self.points_per_rank) // dimensions
        coords_per_rank = self.points_per_rank[:n_coords_per_rank * dimensions]
        coords_per_rank = coords_per_rank.reshape(dimensions, n_coords_per_rank)

        super().__init__(coords_per_rank,self.boundaries)


    def parallel_integrate(self, func):
        "enables each rank to integral with the mean,varience and error(std)"
        local_integral = self.integrate(func)
        
        local_stats = self.mean_var_std(func)

        n_total = len(self.coords)*len(self.coords[0])

        par_integral = self.comm.reduce(local_integral, op = MPI.SUM , root = 0 )

        results = (local_integral, local_stats[0],
                       local_stats[1], local_stats[2])


        expected_val = local_stats[0]
        expected_val_squared = np.mean(self.f_array**2)

        par_expected_val = self.comm.reduce(expected_val,
                                                 op = MPI.SUM, root = 0)

        par_expected_val_squared = self.comm.reduce(expected_val_squared,
                                                 op = MPI.SUM, root = 0)

        if self.rank == 0:

            par_integral = par_integral /self.procs

            boundary_dim = (self.boundaries[1] - self.boundaries[0])**self.dim

            var = 1/n_total *( (par_expected_val_squared/self.procs)
                                   - (par_expected_val/self.procs)**2 )

            error = np.sqrt(var) * boundary_dim
            
            print('\nRank 0 - reduced MPI.SUM result\n-----------------------',
                  f'\n{self.dim} dimentional {func.__name__}',
                  f'\nIntegral = {par_integral}',
                  f'\nMean = {expected_val}',
                  f'\nVar = {var}',
                  f'\nStd = {error}')

            return par_integral, expected_val, var, error
        
        
        
        

        return results






def sin(xvals):
    "simple sin function for test"
    return np.sin(xvals)
def circ(coords):
    "circle function for estimating pi/4"
    rad_point = np.sqrt( np.sum(coords**2, axis = 0) )
    radius = 1
    rad_arr = np.where(rad_point < radius,1,0)
    #ratio = fsum(rad_arr)/len(rad_arr)
    return rad_arr


def gaussian(coords):
    "the Gaussian distribution function"

    sigma  = 1

    x0 =  np.zeros(len(coords[:,0]))
    num_x0 = len(coords[:,0])
    x0 = x0[num_x0-1]

    x_new = coords/(1-coords**2)

    t_coefficient = np.prod((1+coords**2)/(1-coords**2)**2 ,axis =0)


    gauss = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(np.sum(-((x_new - x0) ** 2)
                                                              / (2 * sigma**2), axis =0))

    return  t_coefficient * gauss



if __name__ == "__main__":

    ###########################################################################
    #Initial Conditions
    ###########################################################################
    
    LOW_LIM = -1
    UP_LIM  = 1
    bounds = np.array([LOW_LIM,UP_LIM])

    NUM_PER_RANK = 1000000
    N_DIM = 6

    ###########################################################################
    #Task 1
    ###########################################################################

    'Repeats integral calulation for each Dimention'
    for dim in range(1,N_DIM+1):
        par_circ = ParallelMonteCarlo(NUM_PER_RANK, bounds, dim)
        par_circ_integral = par_circ.parallel_integrate(circ)


    ###########################################################################
    #Task 2
    ###########################################################################
    'Repeats integral calulation for each Dimention'
    for dim in range(1,N_DIM+1):
        par_guass = ParallelMonteCarlo(NUM_PER_RANK, bounds, dim)
        par_guass_integral = par_guass.parallel_integrate(gaussian)



    ###########################################################################
