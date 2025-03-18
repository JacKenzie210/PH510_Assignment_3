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
        self.f_array = func(self.coords)# * (self.boundary[1]-self.boundary[0])**self.dim
        self.mean = np.mean(self.f_array)
        self.var = np.var(self.f_array)
        self.std = np.sqrt(self.var) * (self.boundary[1]-self.boundary[0])**self.dim
        return self.mean, self.var,self.std

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
        return

    def circ_points(self):
        "Used to calculate the ratio of points inside and outside the circle"
        self.radius = (abs(self.boundary[0])+abs(self.boundary[1]))/2

        self.inclosed_points = np.delete(self.coords, np.where(self.coords[0]**2+self.coords[1]**2
                                        > self.radius**2),axis = 1)
        self.out_points = np.delete(self.coords, np.where(self.coords[0]**2+self.coords[1]**2
                                        < self.radius**2),axis = 1)

        self.ratio =  fsum( (np.where(self.coords[0]**2 +self.coords[1]**2< self.radius**2
                              , 1, 0) )/ (len(self.coords[0]))  )
        return

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
        print(f'Rank {self.rank}\n----')

        self.procs = self.comm.Get_size()
        self.n_per_rank = n_per_rank
        self.total_points = self.n_per_rank*self.procs

        self.boundaries = boundaries
        self.points_per_rank  = np.random.uniform(self.boundaries[0],
                                                      self.boundaries[1],
                                                      self.n_per_rank)

        self.n_coords_per_rank = len(self.points_per_rank) // dimensions

        self.coords_per_rank = self.points_per_rank[:self.n_coords_per_rank * dimensions]
        self.coords_per_rank = self.coords_per_rank.reshape(dimensions, self.n_coords_per_rank)

        super().__init__(self.coords_per_rank,self.boundaries)


    def parallel_integrate(self, func):
        "enables each rank to integral with the mean,varience and error(std)"
        local_integral = self.integrate(func)
        local_stats = self.mean_var_std(func)
        self.n_total = len(self.coords)*len(self.coords[0])

        self.par_integral = self. comm.reduce(local_integral, op = MPI.SUM , root = 0 ) 

        results = (local_integral, local_stats[0],
                       local_stats[1], local_stats[2])

        self.expected_val = local_stats[0]
        self.expected_val_squared = np.mean(self.f_array**2)

        self.par_expected_val = self.comm.reduce(self.expected_val, 
                                                 op = MPI.SUM, root = 0)
        
        self.par_expected_val_squared = self.comm.reduce(self.expected_val_squared, 
                                                 op = MPI.SUM, root = 0)

        if self.rank == 0:
            
            self.par_integral = self.par_integral /self.procs
            
            boundary_dim = (self.boundaries[1] - self.boundaries[0])**self.dim
            
            var = 1/self.n_total *( (self.par_expected_val_squared/self.procs)
                                   - (self.par_expected_val/self.procs)**2 )

            error = np.sqrt(var) * boundary_dim
            
            return self.par_integral, self.expected_val, var, error
            
        # self.par_integral = self.comm.reduce(local_integral, op=MPI.SUM, root = 0 )
        # self.par_mean =  self.comm.reduce(local_stats[0], op=MPI.SUM, root = 0 )
        
        # self.par_var =  self.comm.reduce(local_stats[1], op=MPI.SUM, root = 0 )
        # self.par_std =  self.comm.reduce(local_stats[2], op=MPI.SUM, root = 0 )




        # if self.rank == 0:
        #     self.par_integral = self.par_integral/self.procs

        #     self.expected_val = self.par_mean/self.procs
        #     self.expected_val_squared = self.expected_val**2 /self.procs

        #     self.par_varience =( (self.expected_val_squared -(self.expected_val/self.procs)**2 )/
        #                         (self.n_coords_per_rank*self.procs) )
        #     self.error = np.sqrt(self.par_var)

        #     results = self.par_integral, self.expected_val, self.par_varience, self.error

        return results






def sin(xvals):
    "simple sin function for test"
    return np.sin(xvals)
def circ(coords):
    "circle function for estimating pi/4"
    rad_point = np.sqrt( np.sum(coords**2, axis = 0) )
    radius = RAD
    rad_arr = np.where(rad_point < radius,1,0)
    ratio = fsum(rad_arr)/len(rad_arr)
    return rad_arr


def gaussian(coords):
    "the Gaussian distribution function"

    sigma  = 1
    x0 = 0
    x_new = coords/(1-coords**2)

    t_coefficient = np.prod( (1+coords**2)/(1-coords**2)**2 , axis = 0)

    
    gauss = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(np.sum(-((x_new - x0) ** 2) / (2 * sigma**2), axis =0))


    return t_coefficient * gauss



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
    bounds = np.array([LOW_LIM,UP_LIM])

    ###########################################################################
    #Testing for non parallel computations
    ###########################################################################
    # print('\n1D testing')
    # arr_1d = np.array([x_arr])
    # test_1d = MonteCarlo(arr_1d, bounds)
    # I =test_1d.integrate(sin)
    # test_1d.plot1d(sin)
    # print(f'integral check = {I} \nmean, var & std = {test_1d.mean_var_std(sin)}')

    # print('\n2D testing')
    # arr_2d = np.array([x_arr, y_arr])
    # test_2d = MonteCarlo(arr_2d, bounds)

    # test_2d1 = MonteCarlo(arr_2d, bounds)
    # a = test_2d  - test_2d1
    # test_2d.plotcirc()
    # print(f'integral check = {test_2d.integrate(circ)}')
    # print(f'ratio = {test_2d.ratio}, pi = {test_2d.ratio*4}')
    # print(f'mean,var & std = {test_2d.mean_var_std(circ)}')

    ###########################################################################
    #Testing for Parallel Computations
    ###########################################################################
    print(f'\n2D parallel Testing \n-------------------')
    NUM_PER_RANK = 100000
    N_DIM = 6
    
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
    
    ###########################################################################

