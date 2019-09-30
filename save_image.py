# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:09:25 2019

@author: Muhammad
"""
import numpy as np
import matplotlib.pyplot as plt
def save_image(data, cm, fn):
   
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    ax.imshow(data, cmap=cm,vmin=np.min(data),vmax=np.max(data))
    ax.autoscale(False)
    plt.savefig(fn, dpi = height) 
    plt.close()