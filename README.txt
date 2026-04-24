On python 3.8.18, there are problems when assigning (and reassigning) variables using Covariance() directly in the instance creation of FiniteRotation() and EulerVector(). For example:

"""
import sys
sys.path.append(r"C:\Users\nbt571\Documents\C_repos\platekinematics\build\lib.win-amd64-cpython-38")
import platekinematics.pk_structs as pks
"""

# WILL CRASH
fr = pks.FiniteRotation(1.0, 1.0, 1.0, 1.0, pks.Covariance([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
fr = pks.FiniteRotation(1.0, 1.0, 1.0, 1.0, pks.Covariance([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))


# WILL NOT CRASH
cov = pks.Covariance([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
fr = pks.FiniteRotation(1.0, 1.0, 1.0, 1.0, cov)
fr = pks.FiniteRotation(1.0, 1.0, 1.0, 1.0, cov)

