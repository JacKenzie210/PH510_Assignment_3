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
plt.xlim(-15,15)
plt.ylim(-15,15)
plt.axis('square')



theta = np.linspace(0,2*np.pi,1000)

xr = r*np.cos(theta)
yr= r*np.sin(theta)

plt.plot(xr,yr,'orange',0,0,'or')

plt.plot([0,xr[125]],[0,yr[125]], '-->' )


xp = [8,-4,10,1]
yp = [7,3,9,-4]
plt.plot(xp,yp,'gx')

point = [6, 7]
plt.plot(point[0],point[1],'kx')
if (point[0]**2 + point[1]**2 )>= r**2:
    print('point outside circle')
else:
    print('point inside circle')

if -r < point[0] < r and  -r < point[1] < r:
    print("inside the square")
else:
    print('outside/on the square')
    
    




