from skimage.io import imread
from skimage.transform import radon, resize
import warnings
import numpy as np
import pywt
import scipy.interpolate as interpolate
from scipy.optimize import minimize
from scipy.signal import correlate
import time
import math
import sys
import scipy.sparse as sp
import os
import matplotlib.pyplot as plt
import argparse
import pathlib

class Tomograph:
    def __init__(self,file_name,target_size=128,n_theta=50,meas_std=0.1,relative_location=""):
        self.globalprefix = pathlib.Path.cwd() / relative_location
        if not self.globalprefix.exists():
            self.globalprefix.mkdir()
        self.file_name= self.globalprefix / file_name
        if target_size>512:
            raise Exception("Target image dimension are too large (" + str(target_size) + " x " + str(target_size) +")")
        self.dim = target_size
        self.meas_std = meas_std
        img = imread(self.file_name.absolute(),as_gray=True)
        if (img.shape[0]!=img.shape[1]):
            raise Exception("Image is not square")
        self.target_image = resize(img, (self.dim, self.dim), anti_aliasing=False, preserve_range=True,
                            order=1, mode='symmetric')
        self.theta = np.linspace(0., 180., n_theta, endpoint=False)
