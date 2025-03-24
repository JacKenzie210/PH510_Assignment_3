#!/usr/bin/env python3
#This code is licenced with MPL 2.0
"""
Created on Mon Mar 24 15:24:18 2025

@author: jackm
"""
import numpy as np
from mpi4py import MPI
from monte_carlo import MonteCarlo


class ParallelMonteCarlo(MonteCarlo):
    """
    A sub class of MonteCarlo enabling parallel opperations using MPI
    """

    def __init__(self, n_per_rank:int,boundaries, dimensions:int):
        "initial conditions and the start of the parallelism"
        
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
        """enables each rank to integrate the function 
        with the mean,varience and error(std)"""
        
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

    def circ(coords):
        "circle function for estimating pi/4"
        rad_point = np.sqrt( np.sum(coords**2, axis = 0) )
        radius = 1
        rad_arr = np.where(rad_point < radius,1,0)
        return rad_arr


    def gaussian(coords):
        "the Gaussian distribution function"

        sigma  = 1

        #x0 is an array of values the size of number of dimensions which is 
        #currently set to all 0s but can be changed to any set of values.
        x0 =  np.zeros(len(coords[:,0]))
        num_x0 = len(coords[:,0])
        x0 = x0[num_x0-1]

        x_new = coords/(1-coords**2)

        t_coefficient = np.prod((1+coords**2)/(1-coords**2)**2 ,axis =0)


        gauss = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(np.sum(-((x_new - x0) ** 2)
                                                                  / (2 * sigma**2), axis =0))

        return  t_coefficient * gauss

    ###########################################################################
    #Testing for Parallel Computations
    ###########################################################################
    print('\n2D parallel Testing \n-------------------')
    LOW_LIM = -1
    UP_LIM  = 1
    bounds = np.array([LOW_LIM,UP_LIM])

    NUM_PER_RANK = 1000000
    N_DIM = 6
    N_COORDS = NUM_PER_RANK // N_DIM

    ###################
    #circle/sphere etc
    ###################
    test_par = ParallelMonteCarlo(NUM_PER_RANK, bounds, N_DIM)
    test_par_integral = test_par.parallel_integrate(circ)

    print(f'integral = {test_par_integral[0]}' )
    print(f'Mean = {test_par_integral[1]}' )
    print(f'Var = {test_par_integral[2]}' )
    print(f'Std = {test_par_integral[3]}' )

    ###################
    #gaussian
    ###################
    par_guass = ParallelMonteCarlo(NUM_PER_RANK, bounds, N_DIM)
    par_guass_integral = par_guass.parallel_integrate(gaussian)

    print(f'Guassian function of {N_DIM} dimentions')
    print(f'integral = {par_guass_integral[0]}' )
    print(f'Mean = {par_guass_integral[1]}' )
    print(f'Var = {par_guass_integral[2]}' )
    print(f'Std = {par_guass_integral[3]}' )
