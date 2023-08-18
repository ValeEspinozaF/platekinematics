# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 08:42:10 2023

@author: nbt571
"""

from ctypes import *
import platekinematics.build_ensemble as yec

class Covar(Structure):
    _fields_ = [
        ("C11", c_double),
        ("C12", c_double),
        ("C13", c_double),
        ("C22", c_double),
        ("C23", c_double),
        ("C33", c_double)
        ]

class FiniteRotationSph(Structure):
    _fields_ = [
        ("Lon", c_double),
        ("Lat", c_double),
        ("Angle", c_double),
        ("Time", c_double),
        ("Covariance", Covar)
        ]
    
    
cov = Covar(1, 2, 3, 4, 5, 6)
fr = FiniteRotationSph(20, 30, 2, 0.5, cov)




class Mec(Structure):
    _fields_ = [
        ("age", c_int),
        ("number", c_int),
    ]

class Bad(Structure):
    _fields_ = [
        ("age", c_int),
        ("number", c_int),
        ("extra", c_int),
    ]

m = Mec(1, 2)
print(yec.viapoint(m))

# TypeError
b = Bad(1, 2, 3)
print yec.viapoint(b)



#platekinematics.build_ensemble.viapoint(m)