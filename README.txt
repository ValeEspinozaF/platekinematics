On python 3.8.18, there are problems when assigning (and reassigning) variables using Covariance() directly in the instance creation of FiniteRotation() and EulerVector(). For example:

"""
import sys
sys.path.append(r"C:\Users\nbt571\Documents\C_repos\platekinematics\build\lib.win-amd64-cpython-38")
import platekinematics as pk
"""

# WILL CRASH
fr = pk.FiniteRotation(1.0, 1.0, 1.0, 1.0, pk.Covariance([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
fr = pk.FiniteRotation(1.0, 1.0, 1.0, 1.0, pk.Covariance([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))


# WILL NOT CRASH
cov = pk.Covariance([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
fr = pk.FiniteRotation(1.0, 1.0, 1.0, 1.0, cov)
fr = pk.FiniteRotation(1.0, 1.0, 1.0, 1.0, cov)

