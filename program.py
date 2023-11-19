# PROGRAM: BiCRYSTAL

# VERSION: 1.0.8

# DESCRIPTION: This program builds commensurate crystal structures of hexagonal layered materials. Current version reads CIF files and writes the new structure to a QUANTUM ESPRESSO input file.

# AUTHOR: T. Kabengele, Johnson Chemistry Group, Dalhousie University

#********************************************************************************************************

# INITIALIZATION

import sys, csv, os
import pandas
from operator import add
from crystals import Crystal, Atom, Element, distance_fractional, distance_cartesian
from scipy.spatial.distance import cdist, pdist
from shapely.geometry import Point, MultiPoint, Polygon
from shapely.ops import nearest_points
import numpy as np
import decimal
from datetime import date, datetime
import time

# importing helper functions
from helpers_bicrystal import *

#********************************************************************************************************

# MAIN PROGRAM:


# Start...
print ('\n\n*************************** WELCOME TO BiCRYSTAL ***************************\n\n * BiCRYSTAL: Commensurate and incommensurate crystal structures of layered materials\n (c) 2020 Johnson Chemistry Group, Dalhousie University\n')
print (' Description: BiCrystal builds commensurate structures of hexagonal 2D materials.\n Current version reads CIF files and writes the new structure to a QUANTUM ESPRESSO input file.\n Additional information such as the bond distance between atoms, lattice vectors in Bohr and Angstrom, and a simple 3D plot of each layer is also provided.\n\n Authored by: T. Kabengele \n Contact: tilas.kabengele@dal.ca \n\n using: Python 3.7.6 \n with libraries: numpy, matplotlib \n imported library: crystals \n (c) Copyright 2020, Laurent P. René de Cotret \n Install via: pip install crystals\n\n')

now = datetime.now()
print ('* BiCRYSTAL--', now,'\n\n')
print ('**************************************************************************\n \n')

# reading program.py from workspace, i.e. directory where you installed bicrystal
# if bicrsytal directory same as program.py, do not modify*
with open('program.py') as f:
    line = f.readline()

# reading csv with elements from workspace
program_directory = str(line).rstrip("\n")
colnames = ['number', 'symbol']
periodic_table = pandas.read_csv('periodic_table.csv', usecols=colnames)
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

#### Initializing top and bottom layers ####
tt_top,tt_bot,elt_top,elt_bot = bulk(my_crystal)

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
print ( " title='crystal'")
print ( " prefix='crystal'")
print ( " pseudo_dir='.'")
print ( " calculation='relax'\n etot_conv_thr=1.0D-5\n forc_conv_thr=1.0D-4\n/")
print ( "&system")
print (" ibrav=0")
print (" nat=atoms")

nat = ntype(my_crystal)
print (" ntyp= ",nat,"",sep="")
print (" ecutwfc=80.0")
print (" ecutrho=800.0")
print (" vdw_corr= 'xdm'")
print ("/\n&electrons")
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
v1rb,v2rb,v3rb,v4rb = rotcell(v1b*ex,v2b*ex,v3b*ex,v4b*ex,origin,Rb)
boundary_bot,p1b,p2b,p3b,p4b = poly(v1rb,v2rb,v3rb,v4rb)

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

# delete equivalent atoms
if len(sim) >= 1:
    bot_frac = np.delete(bot_frac, sim[1:], 0)
    symbot = np.delete(symbot, sim[1:], 0)

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
v1rt,v2rt,v3rt,v4rt = rotcell(v1t*ex,v2t*ex,v3t*ex,v4t*ex,origin,Rt)
boundary_top,p1t,p2t,p3t,p4t = poly(v1rt,v2rt,v3rt,v4rt)

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

# delete equivalent atoms
if len(sim) >= 1:
    top_frac = np.delete(top_frac, sim[1:], 0)
    symtop = np.delete(symtop, sim[1:], 0)

i = 0
nat_top=0
for atm in top_frac:
    print ('{:2} {:12.6f} {:12.6f} {:12.6f}'.format(symtop[i], atm[0], atm[1], atm[2]))
    i+=1
    nat_top+=1

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

########################################
# Suplemmentary info about moire angles
#***************************************
relative_rotation_angle = round(calculate_relative_rotation_angle(m, n), 2)
moire_lattice_constant = round(np.linalg.norm(newa1),4)

########################## SUMMARY REPORT ###############################

print ("\n********************* SUMMARY REPORT ***********************")
#print ('\nRotation angle (deg) = ', np.round(rotation_angle,3))
#print ('Relative Rotation (deg) = ',np.round(np.rad2deg(phi),3))
print ('\nRelative rotation angle (degrees) = ', relative_rotation_angle)
print ('Moire lattice constant (anstrom) = ', moire_lattice_constant)
print ('\nTop atoms(rotated) = ',len(top_frac))
print ('Bottom atoms  = ',len(bot_frac))
print ('\nTotal atoms \n=', len(bot_frac)+len(top_frac))
#print ('time for replication', elbt1)
#print ('time for plotting', elplt)
#print ('\ntotal time top layer',eltp)
#print ('total time bottom layer',elbt)
print ('\n*************************** Done!! **************************\n')

###################
# OPTIONAL PLOTTING
# ****************** 
import subprocess
import json

# bottom top layer atoms and parameters

coord_b = [p1b, p2b, p3b, p4b]
coord_b.append(coord_b[0])
#coord_b = np.array(coord_b)


zb = np.array(lowest(my_crystal))

#bot = np.array(bot)
#botl = np.array(botl)

coord_t = [p1t, p2t, p3t, p4t]
coord_t.append(coord_t[0])


#coord_t = np.array(coord_t)


zt = np.array(highest(my_crystal))

#top = np.array(top)
#topl = np.array(topl)

atm = np.array(atm)



# converting to list for json compatibility
#******************************************

atm = atm.tolist()

#coord_b = coord_b.tolist()
zb = zb.tolist()
bot = bot.tolist()
#botl = botl.tolist()

#coord_t = coord_t.tolist()
zt = zt.tolist()
top = top.tolist()
#topl = topl.tolist()

v1rb = v1rb.tolist()
v2rb = v2rb.tolist()
v3rb = v3rb.tolist()
v4rb = v4rb.tolist()

v1rt = v1rt.tolist()
v2rt = v2rt.tolist()
v3rt = v3rt.tolist()
v4rt = v4rt.tolist()

# writing to file
# ***************
with open('atomic_plotting_data.json', 'w') as file:
    json.dump({
        'v1rb': v1rb,
        'v2rb': v2rb,
        'v3rb': v3rb,
        'v4rb': v4rb,
        'v1rt': v1rt,
        'v2rt': v2rt,
        'v3rt': v3rt,
        'v4rt': v4rt,
        'atm': atm,
        'zb': zb,
        'zt': zt,
        'bond_distance': bond_distance,
        'bot': bot,
        'symb_num_bot': symb_num_bot,
        'symb_num_top': symb_num_top,
        'botl': botl,
        'top': top,
        'topl': topl,
        'coord_b': coord_b,
        'coord_t': coord_t
    }, file)

#*******************************************************************
max_attempts = 3
attempt = 1
while attempt <= max_attempts:
    view_structure = input("\nDo you want to generate a plot? [Y/n]: ")

    if view_structure.lower() == "y":
        # Run plotting_bicrystal.py subroutine as a python subprocess
        # ***********************************************************
        print("\nPlotting...")
        subprocess.call(['python', 'plotting_bicrystal.py', file.name])
        break
    elif view_structure.lower() == "n":
        print("\nNo worries! Moving on...")
        break
    else:
        print("\nInvalid input! Please enter either 'y' or 'n'.")
        attempt += 1

if attempt > max_attempts:
    print("\nMaximum attempts reached. Exiting...")
#*********************************************************************
