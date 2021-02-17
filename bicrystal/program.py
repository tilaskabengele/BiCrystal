#!/bin/python3

# PROGRAM: BiCRYSTAL

# VERSION: 1.0.8

# DESCRIPTION: This program buildscommensurate and incommensurate crystal structures of layered materials. Current version reads CIF files and writes the new structure to a QUANTUM ESPRESSO input file.

# AUTHOR: T. Kabengele, Johnson Chemistry Group, Dalhousie University

# REQUIREMENTS:
# Python "crystals" package must be installed on your system
# Install via: pip install crystals
# Alternative install: conda install -c conda-forge crystals

#********************************************************************************************************

# INITIALIZATION

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
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import path
from datetime import date, datetime
import time


# FUNCTIONS:

# ***********************************
# Function Name: SEPERATE(MY_CRYSTAL)
# Description: Isolates the z-coordinates (fractional) from CRYSTAL and stores them in an array
# Inputs: Python CRYSTAL structure. i.e. don't use a supercell in SEPERATE
# Output: Array with z-coordinates of bottom layer atoms

def seperate(my_crystal):

        nat = 0
        z_frac = []

        for atm in my_crystal:
                nat = nat + 1
                z_frac.append(atm.coords_fractional[2])

        z_sorted = sorted(z_frac)
        bottom_z = round(nat/2)
        answer = z_sorted[0:bottom_z]

        return answer


# ***********************************
# Function Name: LOWEST(MY_CRYSTAL)
# Description: Finds the lowest z position in the crystal
# Inputs: Python CRYSTAL structure. i.e. don't use a supercell
# Output: z coordinate in angstrom of lowest atom

def lowest(my_crystal):

        nat = 0
        z_cart = []

        for atm in my_crystal:
                nat = nat + 1
                z_cart.append(atm.coords_cartesian[2])

        return min(z_cart)



# ***********************************
# Function Name: HIGHEST(MY_CRYSTAL)
# Description: Finds the highest z position in the crystal
# Inputs: Python CRYSTAL structure. i.e. don't use a supercell
# Output: z coordinate in angstrom of highest atom

def highest(my_crystal):

        nat = 0
        z_cart = []

        for atm in my_crystal:
                nat = nat + 1
                z_cart.append(atm.coords_cartesian[2])

        return max(z_cart)


#**************************************
# Function name: LOCATE(TEST_ATM,Z_BOT)
# Description: Checks if a given atom is from the bottom or top layer
# Inputs: The z-coordinate of a trial atom; an array with z-coordinates of bottom layer atoms
# Output: True if atom is from bottom layer; false if it isn't

def locate(test_atm,z_bot):

        z = []

        test_atm = round(test_atm,2)

        for atm in z_bot:
                z.append(round(atm,2))

        if test_atm in z:
                answer = True
        else:
                answer = False

        return answer


# ***********************************
# Function Name: INTERLAYER(atm_top,atm_bottom)
# Description: Finds the interlayer seperation bettwen the layers
# Inputs: Python crystal
# Output: Interlayer seperation in bohr

def interlayer(my_crystal):

        atmz = []
        for atm in my_crystal:
            atmz.append(atm.coords_cartesian[2])

        z = sorted(atmz)
        z_bot = z[:len(z)//2]
        z_top = z[len(z)//2:]
        seperation = min(z_top) - max(z_bot)

        return seperation*0.529177249



#***********************************
# Function name: NEWCELL(MY_CRYSTAL,M,N)
# Description: This function creates lattice vectors of a rotated layer
# Inputs: Python crystal; expansion parameters of desired supercell
# Outputs: supercell lattice vectors new_a; new_b .i.e the c-vector remains unchanged

def newcell(my_crystal,atoms,m,n):

#        a, b, c = my_crystal.lattice_vectors
        a1, a2, a3, alpha, beta, gamma = my_crystal.lattice_parameters
#        v1 = atoms[0] + (m+n)*a + (m+n)*b
#        v2 = np.add(np.add(v1,(m+n)*a),n*b)
#        v3 = np.add(np.add(v2,(m+n)*b),m*a)
#        v4 = np.subtract(np.subtract(v3,(m+n)*a),n*b)

        old_a = np.array([0.5*a1*np.sqrt(3), -0.5*a2, 0])
        old_b = np.array([0, a2, 0])

        v1 = atoms[0] + (m+n)*old_a + (m+n)*old_b
        v2 = np.add(np.add(v1,(m+n)*old_a),n*old_b)
        v3 = np.add(np.add(v2,(m+n)*old_b),m*old_a)
        v4 = np.subtract(np.subtract(v3,(m+n)*old_a),n*old_b)

#        old_a = np.array([a1*0.5*np.sqrt(3), -0.5*a2, 0])
#        old_b = np.array([0, a2, 0])
#        old_c = np.array([0, 0, a3])

#        new_a = n*old_b - m*old_a
#        new_b = m*old_b - n*old_a

        new_a = v3 - v2
        new_b = v2 - v1

        # ang to bohr mytiply by 1.8897259886

        return new_a, new_b, v1, v2, v3, v4




#***********************************
# Function name: ROTCELL(V1,V2,V3,V4,ORIGIN)
# Description: This function gives the rotated coordinates lattice vertices of newcell
# Inputs: vertices, origin atom, rotation matrix
# Outputs: rotated vertices

def rotcell(v1,v2,v3,v4,origin,R):
    v1 = v1 - origin
    v2 = v2 - origin
    v3 = v3 - origin
    v4 = v4 - origin
    vr1 = np.dot(v1,R)
    vr2 = np.dot(v2,R)
    vr3 = np.dot(v3,R)
    vr4 = np.dot(v4,R)

    return vr1,vr2,vr3,vr4



#***************************************
# Function name: PLOY(V1,V2,V3,V4)
# Description: Create polygon / new unit cell from vertices
# Input: Vertices of the supercell
# Output: polygon

def poly(v1,v2,v3,v4):

        p1=v1[0], v1[1]
        p2=v4[0], v4[1]
        p3=v3[0], v3[1]
        p4=v2[0], v2[1]

        coords = [(p1), (p2), (p3), (p4)]
        ply = Polygon(coords)

        return ply, p1, p2, p3, p4




#***************************************
# Function name: INPLOY(ATZ,POLY)
# Description: Determine if an atom lies within super unit cell
# Input: Vertices of the super unit cell
# Output: TRUE, if atom is in, FALSE if not

def inpoly(atz,ply):

        atm = Point(atz[0], atz[1])
        check1 = atm.within(ply)
        check2 = atm.intersects(ply)
        if check1 == True:
                return check1
        if check2 == True:
                return check2


#*****************************************
# Function name: CENTRAL(ATM,V1,V2,V3,V4)
# Description: Determine the centre of unit cell
# Input: Vertices of the super unit cell
# Output: central coordinates of unit cell

def central(ply):

        center = ply.centroid

        return center


#***************************************
# Function name: SWAPED(LAYER)
# Description: Check if rotation was performed correctly
# Input: Array with rotated atoms
# Output: True if the rotation is correct, False if it isn't

def swaped(layer):
        pos_count = 0
        neg_count = 0
        for atm in layer:
                if atm >= 0:
                        pos_count += 1
                else:
                        neg_count += 1
        if pos_count > neg_count:
                return True
        else:
                return False


#***************************************
# Function name: NTYPE(MY_CRYSTAL)
# Description: Check how many atomic species present in crystal
# Input: python Crystal
# Output: Number of atoms ---> Integer

def ntype(my_crystal):
        nat = 0
        for atm in my_crystal.chemical_composition:
                nat+=1
        return nat



#***************************************
# Function name: BULK(MY_CRYSTAL)
# Description: Create arrays of top and bottom atoms in the crystal
# Input: python Crystal
# Output: top and bottom atoms ---> np arrays; chemical symbols for top and bottom  atoms ---> lists

def bulk(my_crystal):

        atoms_bot = []
        atoms_top = []
        atmxyz = []
        ele_bot = []
        ele_top = []
        z_bot = seperate(my_crystal)
        for atm in my_crystal:
                atm_frac = atm.coords_fractional
                atm_cart = atm.coords_cartesian
                atmxyz = atm_cart[0], atm_cart[1], atm_cart[2]
                if locate(atm_frac[2],z_bot) == True:
                        atoms_bot.append(atmxyz)
                        ELE = str(atm).lower()
                        ele_bot.append(ELE)
                else:
                        atoms_top.append(atmxyz)
                        ELE = str(atm).lower()
                        ele_top.append(ELE)
        return np.array(atoms_top), np.array(atoms_bot), ele_top, ele_bot




#********************************************************************************************************

# MAIN PROGRAM:


# Start...
print ('\n\n*************************** WELCOME TO BiCRYSTAL ***************************\n\n * BiCRYSTAL: Commensurate and incommensurate crystal structures of layered materials\n (c) 2020 Johnson Chemistry Group, Dalhousie University\n')
print (' Description: BiCrystal builds commensurate and incommensurate structures of layered materials.\n Current version reads CIF files and writes the new structure to a QUANTUM ESPRESSO input file.\n Additional information such as the bond distance between atoms, lattice vectors in Bohr and Angstrom, and a simple 3D plot of each layer is also provided.\n\n Authored by: T. Kabengele \n Contact: tilas.kabengele@dal.ca \n\n using: Python 3.7.6 \n with libraries: numpy, matplotlib \n imported library: crystals \n (c) Copyright 2020, Laurent P. René de Cotret \n Install via: pip install crystals\n\n')

now = datetime.now()
print ('* BiCRYSTAL--', now,'\n\n')
print ('**************************************************************************\n \n')

# reading csv with elements from workspace
# i.e. directory where you installed bicrystal
with open('workspace') as f:
    line = f.readline()
program_directory = str(line).rstrip("\n")
dir1 = os.path.join("" +program_directory+"", 'periodic_table.csv')
colnames = ['number', 'symbol']
periodic_table = pandas.read_csv(dir1, usecols=colnames)
number = periodic_table.number.tolist()
symbol = periodic_table.symbol.tolist()

# Input crystal
my_crystal = Crystal.from_cif(input('***Input cif file*** \n'))
#print (my_crystal)

# Input super cell parameters and rotation angle
print ('\n***Rotation parameters*** ')
m = int(input('Enter m '))
n = int(input('Enter n '))
#rotation_angle = float(input('Enter rotation angle '))

# lattice parameters
a, b, c, alpha, beta, gamma = my_crystal.lattice_parameters

# specify vacuum size in bohr
#print('\n***Specify z-coordinate in cell parameter***')
#print ('(Default = ', c*1.8897259886,' Bohr)')
#vacuum = float(input('Enter z value [Bohr] '))

#### Initializing top and bottom layers ####
tt_top,tt_bot,elt_top,elt_bot = bulk(my_crystal)

# Interlayer seperation #
#d = interlayer(my_crystal)

# lattice vectors
a1, a2, a3 = my_crystal.lattice_vectors
uc = a1,a2,a3
uc = np.array(uc)

print ('\n\nIntializing atoms...\n\n')

print ('Initial TOP atoms..')
for i in range(0,len(elt_top)):
    s = np.array(tt_top[i])
    s = np.dot(s, np.linalg.inv(uc))
    print ('Atom No.',i+1, ' ',elt_top[i], ' ',s)
print ('\nInitial BOTTOM atoms..')
for j in range(0,len(elt_bot)):
    s = np.array(tt_bot[j])
    s = np.dot(s, np.linalg.inv(uc))
    print ('Atom No.',len(elt_top)+j+1, ' ',elt_bot[j], ' ',s)


# selecting zeroeth atoms from t/b layers
print ('\nSelect zeroeth TOP atom')
zeroeth1 = int(input('Enter Atom No. '))
print ('\nSelect zeroeth BOTTOM atom')
zeroeth2 = int(input('Enter Atom No. '))

# correct array indices for zeroeth atoms
idx1 = zeroeth1-1
idx2 = (zeroeth2-1)-len(elt_top)
print ('\nZeroeth TOP (angstrom)', elt_top[idx1], tt_top[idx1])
print ('\nZeroeth BOTTOM (angstrom)', elt_bot[idx2], tt_bot[idx2])

# finding the bond length
lengths = pdist(tt_bot, 'euclidean')
bond_distance = round(min(lengths),3)
print ('\nBond distance = ', bond_distance)

# lattice parameters
print ('\nLattice Vectors (Angstrom)')
print (' ','{:12.6f} {:12.6f} {:12.6f}'.format(a1[0],a1[1],a1[2]))
print (' ','{:12.6f} {:12.6f} {:12.6f}'.format(a2[0],a2[1],a2[2]))
print (' ','{:12.6f} {:12.6f} {:12.6f}'.format(a3[0],a3[1],a3[2]))
a2b = 1.8897259886
print ('\nLattice Vectors (Bohr)')
print (' ','{:12.6f} {:12.6f} {:12.6f}'.format(a1[0]*a2b,a1[1]*a2b,a1[2]*a2b))
print (' ','{:12.6f} {:12.6f} {:12.6f}'.format(a2[0]*a2b,a2[1]*a2b,a2[2]*a2b))
print (' ','{:12.6f} {:12.6f} {:12.6f}'.format(a3[0]*a2b,a3[1]*a2b,a3[2]*a2b))

# rotation angle
A = np.array([1,0,0])
B = np.array([np.cos(np.deg2rad(60)),np.sin(np.deg2rad(60)),0])
V = m*A + n*B
rotation_angle = np.arccos((np.dot(A,V.T))/(np.linalg.norm(A)*np.linalg.norm(V)))
rotation_angle = np.rad2deg(rotation_angle)

# Rotation matrix
theta = np.deg2rad(rotation_angle)
phi = np.deg2rad(60) - 2*theta
R = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
print ('\nRotation angle theta (degrees) = ', rotation_angle)
print ('\nMoire angle gamma (degrees) = ',np.rad2deg(phi))

print ("\n\nCALCULATING ATOMIC POSITIONS...")
for i in range(1,2):
        print ('\n\nPlease wait...\n\n')
        sys.stdout.flush()

print ("&control")
print ( " title='crystal',")
print ( " prefix='crystal',")
print ( " pseudo_dir='.',")
print ( " calculation='relax',\n etot_conv_thr=1.0D-5,\n forc_conv_thr=1.0D-4,\n/")
print ( "&system")
print (" ibrav=0,")
print (" nat= atoms,")

nat = ntype(my_crystal)
print (" ntyp= ",nat,",",sep="")
print (" ecutwfc=80.0,")
print (" ecutrho=800.0,")
print (' xdm=.true.,')
print (" xdm_a1=0.6512,")
print (" xdm_a2=1.4633,\n/")
print ("&electrons")
print (" conv_thr = 1d-8\n/\n&ions\n/\n&cell\n/")
print ("ATOMIC_SPECIES")

for atm in my_crystal.chemical_composition:
        ELE = str(atm)
        ele = ELE.lower()
        mass_number = Element(ele).mass
        print (ele, "     ",mass_number," ",ele,'.UPF',sep="")


print('\nATOMIC_POSITIONS crystal')
#print('\nATOMIC_POSITIONS angstrom')

################### loops for bottom layer ##################
#############################################################

bt0 = time.time()
bt1 = time.time()


tt1 = []
tt2 = []
atoms_bot = []
symb1 = []
symb2 = []
k = 0
for atm in tt_bot:
    u = elt_bot[k]
    k = k + 1
    for i in range(0,(m+n)*15):
        ttx = atm + i*a1
        tt1.append(ttx)
        symb1.append(u)
k = 0
for atm in tt1:
    u = symb1[k]
    k = k + 1
    for i in range(1,(n+m)*15):
        tty = atm + i*a2
        tt2.append(tty)
        symb2.append(u)
symb_bot = symb1 + symb2
atoms_bot = list(tt1) + list (tt2)
#atoms_bot = np.array(atoms_bot)

elbt1 = time.time() - bt1

### Initializing new unit cell ###

# new cell parameters
newa1b, newa2b, v1b, v2b, v3b, v4b = newcell(my_crystal,tt_bot[idx2],m,n)
unitcell = newa1b,newa2b,a3
unitcell = np.array(unitcell)

# polygon boundary
boundary_bot,p1b,p2b,p3b,p4b = poly(v1b,v2b,v3b,v4b)

# center atom
org = central(boundary_bot)
destinations = MultiPoint(atoms_bot)
nearest_geoms = nearest_points(org, destinations)
origin = np.array([nearest_geoms[1].x, nearest_geoms[1].y, 0])
#origin = 0,0,0

ex = 1
Rb = np.array([[np.cos(0), -np.sin(0), 0], [np.sin(0), np.cos(0), 0], [0, 0, 1]])
v1r,v2r,v3r,v4r = rotcell(v1b*ex,v2b*ex,v3b*ex,v4b*ex,origin,Rb)
boundary_bot,p1b,p2b,p3b,p4b = poly(v1r,v2r,v3r,v4r)

# list of atomic numbers from symbols
symb_num_bot = []
for i in range(0,len(symb_bot)):
    idx = symbol.index(symb_bot[i])
    symb_num_bot.append(number[idx])

# number of types of atomic species
typ = ntype(my_crystal)

# Initializing check for which atoms lie within the new unit cell
bot = []
symbot = []
atoms_bot = atoms_bot - origin
supx,supy,supz = atoms_bot.T


for i in range(0,len(atoms_bot)):
    num = symb_num_bot[i]
    if inpoly(atoms_bot[i],boundary_bot) == True:
        bt =  supx[i], supy[i], supz[i]
        bot.append(bt)
        symbot.append(symb_bot[i])

botl = bot
bot = np.array(bot)
#bot = bot - origin
bot_frac = np.dot(bot, (np.linalg.inv(unitcell)))

for i in range(1,m+n):
    bot_frac[bot_frac<0] += 1
bot_frac[bot_frac>1] += -1

i = 0
sim = []
for atm1 in bot_frac:
    count = 0
    for atm2 in bot_frac:
        if (round(atm1[0],2) == round(atm2[0],2) and round(atm1[1],2) == round(atm2[1],2) and round(atm1[2],2)== round(atm2[2],2)) == True:
            count+=1
        if count > 1:
            sim.append(i)
    i+=1
sim = np.array(sim)
sim = np.unique(sim)
#if len(sim) > 2:
#    bot_frac = np.delete(bot_frac, sim[1:], 0)
#    symbot = np.delete(symbot, sim[1:], 0)
#if len(sim) == 2:
#    bot_frac = np.delete(bot_frac, sim[0:], 0)
#    symbot = np.delete(symbot, sim[0:], 0)
if len(sim) >= 1:
    bot_frac = np.delete(bot_frac, sim[1:], 0)
    symbot = np.delete(symbot, sim[1:], 0)

#print ('bottom atoms angstrom')
#i = 0
#for atm in bot:
#    print ('{} {:12.6f} {:12.6f} {:12.6f}'.format(symbot[i], atm[0], atm[1], atm[2]))
#    i+=1
#print ('bottom atoms crystal')
i = 0
nat_bot=0
for atm in bot_frac:
    print ('{:2} {:12.6f} {:12.6f} {:12.6f}'.format(symbot[i], atm[0], atm[1], atm[2]))
    i+=1
    nat_bot+=1

elbt = time.time() - bt0

#0.5-(atm[2]*c*1.8897259886)/vacuum

####################### loops for top layer ##################
##############################################################
tp0 = time.time()

tt1 = []
tt2 = []
atoms_top = []
symb1 = []
symb2 = []
k = 0
for atm in tt_top:
    u = elt_top[k]
    k = k + 1
    for i in range(0,(m+n)*15):
        ttx = atm + i*a1
        tt1.append(ttx)
        symb1.append(u)
k = 0
for atm in tt1:
    u = symb1[k]
    k = k + 1
    for i in range(1,(n+m)*15):
        tty = atm + i*a2
        tt2.append(tty)
        symb2.append(u)
symb_top = symb1 + symb2
atoms_top = list(tt1) + list (tt2)
#atoms_top = np.array(atoms_top)


### Initializing new unit cell ###

# new cell parameters
newa1t, newa2t, v1t, v2t, v3t, v4t = newcell(my_crystal,tt_top[idx1],m,n)
unitcell2 = newa1t,newa2t,a3
unitcell2 = np.array(unitcell2)

# polygon boundary
boundary_top,p1t,p2t,p3t,p4t = poly(v1t,v2t,v3t,v4t)

# center atom
org = central(boundary_top)
destinations = MultiPoint(atoms_top)
nearest_geoms = nearest_points(org, destinations)
origin = np.array([nearest_geoms[1].x, nearest_geoms[1].y, 0])
#origin = 0,0,0

ex = 1
Rt = Rb
v1r,v2r,v3r,v4r = rotcell(v1t*ex,v2t*ex,v3t*ex,v4t*ex,origin,Rt)
boundary_top,p1t,p2t,p3t,p4t = poly(v1r,v2r,v3r,v4r)

# list of atomic numbers from symbols
symb_num_top = []
for i in range(0,len(symb_top)):
    idx = symbol.index(symb_top[i])
    symb_num_top.append(number[idx])

# number of types of atomic species
typ = ntype(my_crystal)

# Initializing check for which atoms lie within the new unit cell
top = []
symtop = []
atoms_top = np.dot((np.array(atoms_top)-origin), R)
supx,supy,supz = atoms_top.T


for i in range(0,len(atoms_top)):
    num = symb_num_top[i]
    if inpoly(atoms_top[i],boundary_top) == True:
        tt =  supx[i], supy[i], supz[i]
        top.append(tt)
        symtop.append(symb_top[i])

topl = top
top = np.array(top)
#top = top - origin
top_frac = np.dot(top, (np.linalg.inv(unitcell2)))

# removing negative frac coordinates and those greater than 1
for i in range(1,m+n):
    top_frac[top_frac<0] += 1
top_frac[top_frac>1] += -1

i = 0
sim = []
for atm1 in top_frac:
    count = 0
    for atm2 in top_frac:
        if (round(atm1[0],2) == round(atm2[0],2) and round(atm1[1],2) == round(atm2[1],2) and round(atm1[2],2)== round(atm2[2],2)) == True:
            count+=1
        if count > 1:
            sim.append(i)
    i+=1
sim = np.array(sim)
sim = np.unique(sim)
#if len(sim) > 2:
#    top_frac = np.delete(top_frac, sim[1:], 0)
#    symtop = np.delete(symtop, sim[1:], 0)
#if len(sim) == 2:
#    top_frac = np.delete(top_frac, sim[0:], 0)
#    symtop = np.delete(symtop, sim[0:], 0)
if len(sim) >= 1:
    top_frac = np.delete(top_frac, sim[1:], 0)
    symtop = np.delete(symtop, sim[1:], 0)

#print ('top atoms angstrom')
#i = 0
#for atm in top:
#    print ('{} {:12.6f} {:12.6f} {:12.6f}'.format(symtop[i], atm[0], atm[1], atm[2]))
#    i+=1
#print ('top atoms crystal')
i = 0
nat_top=0
for atm in top_frac:
    print ('{:2} {:12.6f} {:12.6f} {:12.6f}'.format(symtop[i], atm[0], atm[1], atm[2]))
    i+=1
    nat_top+=1

#0.5+(atm[2]*c*1.8897259886)/vacuum
eltp = time.time() - tp0
########################################################
############## closing part of scf.in file #############


# k_points
print ('\nK_POINTS automatic')
print ('8 8 1 1 1 1')

#new_a = np.linalg.norm(newa1b)
old_a = np.array([a*0.5*np.sqrt(3), -0.5*b, 0])
old_b = np.array([0, b, 0])
old_c = np.array([0, 0, c])

newa1 = -m*old_a + n*old_b
newa2 = -m*old_b + n*old_a

# cell parameters in bohr
uc1 = np.around(newa1b*1.8897259886, decimals=12)
uc2 = np.around(newa2b*1.8897259886, decimals=12)
uc3 = np.around(a3*1.8897259886, decimals=12)
uc1 = list(uc1)
uc2 = list(uc2)
uc3 = list(uc3)

#unit_cell = ' '.join([str(elem) for elem in uc])
print ('\nCELL_PARAMETERS bohr')
print ('   ','{:17.12f} {:17.12f} {:17.12f}'.format(uc1[0],uc1[1],uc1[2]))
print ('   ','{:17.12f} {:17.12f} {:17.12f}'.format(uc2[0],uc2[1],uc2[2]))
print ('   ','{:17.12f} {:17.12f} {:17.12f}'.format(uc3[0],uc3[1],uc3[2]))

#############################################################
##################### PLOTTING RESULTS ######################

pt1 = time.time()

## figure settings ##
fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(111, projection='3d')
fig.set_facecolor('white')
ax.set_facecolor('white')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.axis('off')
plt.grid(b=None)
#ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
#ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
#ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))


# Initializing bottom coordinates for new unitcell plotting
coord = [p1b, p2b, p3b, p4b]
coord.append(coord[0])
xb, yb = zip(*coord)

# plotting boundary at lowest z position
zb = lowest(my_crystal)
plt.plot(xb,yb,zb)

# plotting bottom layer atoms
bot = bot.T
supx,supy,supz = list(bot)
ax.scatter(supx, supy, supz, s=5*num, c='tab:green', alpha=0.4)
ax.scatter(supx, supy, supz, c='tab:blue')

# Adding bonds to bottom layer

neigh = NearestNeighbors(radius=bond_distance+0.1).fit(botl)
for atm in botl:
    rng = neigh.radius_neighbors([list(atm)])
    dis = np.asarray(rng[0][0])
    idx = np.asarray(rng[1][0])
    if inpoly(atm,boundary_bot) == True:
        for i in idx:
            atz = []
            atz.append(atm)
            atz.append(list(botl[i]))
            xb, yb, zb = zip(*atz)
            num = symb_num_bot[i]
            plt.plot(xb,yb,zb, '-bo',alpha=1)


# Initializing TOP coordinates for new unitcell plotting
coord = [p1t, p2t, p3t, p4t]
coord.append(coord[0])
xt, yt = zip(*coord)

# plotting the boundary at hiest z position
zt = highest(my_crystal)
plt.plot(xt,yt,zt)

# plotting TOP layer atoms
top = top.T
supx,supy,supz = list(top)
ax.scatter(supx, supy, supz, s=5*num, c='tab:red', alpha=0.4)
ax.scatter(supx, supy, supz, c='tab:blue')

# Adding bonds to TOP layer

neigh = NearestNeighbors(radius=bond_distance+0.1).fit(topl)
for atm in topl:
    rng = neigh.radius_neighbors([list(atm)])
    dis = np.asarray(rng[0][0])
    idx = np.asarray(rng[1][0])
    if inpoly(atm,boundary_top) == True:
        for i in idx:
            atz = []
            atz.append(atm)
            atz.append(list(topl[i]))
            xt, yt, zt = zip(*atz)
            num = symb_num_top[i]
            plt.plot(xt,yt,zt, '-ro',alpha=1)



plt.show()

elplt = time.time() - pt1


j2 = 90 + np.rad2deg(phi/2)
mk = 1/(2*(np.abs(np.sin((phi/2)))))
########################## SUMMARY REPORT ###############################

print ("\n********************* SUMMARY REPORT ***********************")
#print ('\nRotation angle (deg) = ', np.round(rotation_angle,3))
#print ('Relative Rotation (deg) = ',np.round(np.rad2deg(phi),3))
print ('\nHermann moire rotation = ', j2)
print ('Hermann moire constant = ', mk)
print ('\nTop atoms(rotated) = ',len(top_frac))
print ('Bottom atoms  = ',len(bot_frac))
print ('\nTotal atoms \n=', len(bot_frac)+len(top_frac))
#print ('\n Gamma = ', j2 + np.round(rotation_angle,3))
#print ( '\n lattice vectors = ',1.8897259886*a, 1.8897259886*b, 1.8897259886*c)
#print ('\n Erin method lattice vectors = ',1.8897259886*old_a,1.8897259886*old_b)
#print ('time for replication', elbt1)
#print ('time for plotting', elplt)
#print ('\ntotal time top layer',eltp)
#print ('total time bottom layer',elbt)
print ('\n*************************** Done!! **************************\n')
