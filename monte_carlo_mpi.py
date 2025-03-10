# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 14:45:26 2025

@author: jackm
"""

import numpy as np
import matplotlib.pyplot as plt
import mpi4py as MPI
from monte_carlo import MontiCarlo

class MontiCarloMPI:
    def __innit__(self,coords):
        MontiCarlo(coords)
