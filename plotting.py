# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 18:39:19 2025

@author: jackm
"""

import matplotlib.pyplot as plt
import numpy as np 

r = 10
x,y = r,r
square = [ [-x, -x, x, x, -x] ,[-y, y , y, -y, -y] ]
plt.figure()
plt.plot(square[0],square[1])
plt.xlim(-11,11)
plt.ylim([-11,11])
plt.axis('square')



theta = np.linspace(0,2*np.pi,1000)

xr = r*np.cos(theta)
yr= r*np.sin(theta)

plt.plot(xr,yr,'orange',0,0,'or')

plt.plot([0,xr[125]],[0,yr[125]], '-->' )

point = np.sqrt(xr[125]**2+yr[125]**2)