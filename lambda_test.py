# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 14:37:42 2025

@author: jackm
"""
import numpy as np
import matplotlib.pyplot as plt
class MC:
    
    def __init__(self, x):
        
        self.x = x
        
    def function(self,func):
        
        y = func(self.x)
        
        return y 

def yfunc(x,phi):
    
    return np.sin(x+phi )

xarr = np.linspace(0,2*np.pi, 1000)
                
a = MC(xarr)
phival = np.pi 
        
b = a.function(lambda x: yfunc(xarr,phival))

plt.plot(xarr,b)