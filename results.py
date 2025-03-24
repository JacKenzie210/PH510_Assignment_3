#!/usr/bin/env python3
#This code is licenced with MPL 2.0
"""
Created on Mon Mar 24 15:28:18 2025

@author: jackm
"""

import numpy as np
from mpi4py import MPI
from parallel_monte_carlo import ParallelMonteCarlo



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

if __name__ == "__main__":

    ###########################################################################
    #Initial Conditions
    ###########################################################################

    LOW_LIM = -1
    UP_LIM  = 1
    bounds = np.array([LOW_LIM,UP_LIM])

    NUM_PER_RANK = int(1000000)
    N_DIM = 6
    N_COORDS = NUM_PER_RANK // N_DIM


    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f' {NUM_PER_RANK} points per rank in {N_DIM} dimentions gives',
              f' \n{N_COORDS} coordiantes per rank')


    ###########################################################################
    #Task 1
    ###########################################################################

    for dim in range(1,N_DIM+1):
        par_circ = ParallelMonteCarlo(NUM_PER_RANK, bounds, dim)
        par_circ_integral = par_circ.parallel_integrate(circ)


    ###########################################################################
    #Task 2
    ###########################################################################

    for dim in range(1,N_DIM+1):
        par_guass = ParallelMonteCarlo(NUM_PER_RANK, bounds, dim)
        par_guass_integral = par_guass.parallel_integrate(gaussian)

    ###########################################################################
