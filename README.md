# BiCrystal
``BiCrystal`` is a Python program that builds commensurate and incommensurate crystal structures of layered materials. The current version reads CIF files and writes the new structure to a QUANTUM ESPRESSO input file. The program also provides additional information such as the bond distance between atoms, lattice vectors in Bohr and Angstrom, and a simple 3D plot of each layer.

Contents
==========
    Overview
    Download
    Packages
    Files
    Usage
    Examples
    References
    License

# Overview
Building unit cells of arbitrary size is often an inevitable task when studying the physical and mechanical properties of layered materials such as graphene, hexagonal Boron Nitride, transition metal dichalcogenides. Although most visualzation software such as Xcrysden, VESTA or Avogadro provide very powerful tools for analysing and manipulating periodic crystal structures, constructing large unit cells in bilayers with one of the layers perturbed can be very daunting. ``BiCrystal`` provides a convinient and easy way of creating new crystal structures of arbitrary size from CIF files.

# Download
The latest version of ``BiCrystal`` can be found on github:

https://github.com/tilaskabengele/BiCrystal/


**Contact**: Tilas Kabengele tilas.kabengele@dal.ca

# Packages
`BiCrystal` is a python-based program that uses the following standard packages:
 
    import sys, csv, os
    import os.path
    import pandas
    from operator import add
    from crystals import Crystal, Atom, Element, distance_fractional, distance_cartesian
    from shapely.geometry import Point, MultiPoint,  Polygon
    from scipy.spatial import distance
    from scipy.spatial.distance import cdist, pdist
    from shapely.ops import nearest_points
    from sklearn.neighbors import NearestNeighbors
    from sklearn import neighbors
    from sklearn.neighbors import KNeighborsRegressor
    import numpy as np
    import decimal
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib.collections import PolyCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib import path
    from datetime import date, datetime

