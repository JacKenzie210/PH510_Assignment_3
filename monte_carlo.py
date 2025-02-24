#This code is licenced with MPL 2.0 
"""
Created on Mon Feb 24 2025

@author: jackm
"""

import numpy as np


class monti_carlo:
    
    def __init__(self):
        
        return
    
    def factorial(num):
        result = 1
        for i in range(2, num+1):
            result *= num
        
        return result
    
    def choose(n,k):
        c_n = factorial(n)/(factorial(k)* factorial(n - k))
        return c_n 



