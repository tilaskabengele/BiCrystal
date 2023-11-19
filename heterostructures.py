# This is a sub-module under the bicrystal program that handles building
# heterostructures consisting of an overlayer and a substrate
# **********************************************************************

# REQUIRED PACKAGES
# *****************
import sys, csv, os
import pandas as pd
from operator import add
from crystals import Crystal, Atom, Element, distance_fractional, distance_cartesian
from scipy.spatial.distance import cdist, pdist
from scipy.spatial.transform import Rotation
from shapely.geometry import Point, MultiPoint, Polygon
from shapely.ops import nearest_points
import numpy as np
import decimal
from datetime import date, datetime
import time

# importing helper functions
from helpers_bicrystal import *

print ('\n\n*********************** WELCOME TO BiCRYSTAL-HETEROSTRUCTURES ***************************\n\n *       BiCRYSTAL: Building unit cells of heterostructured materials.\n            (c) 2023  Rubenstein Research Group, Brown University.\n')

now = datetime.now()
print (' *       BiCRYSTAL--', now,'\n')
print ('******************************************************************************************\n\n')

############################# INITIALIZATION ###############################

# reading program.py from workspace, i.e. directory where you installed bicrystal
# if bicrsytal directory same as heterostructures.py, do not modify*
with open('heterostructures.py') as f:
    line = f.readline()

# reading csv with elements from workspace
program_directory = line.rstrip("\n")
colnames = ['number', 'symbol']
periodic_table = pd.read_csv('periodic_table.csv', usecols=colnames)
number = periodic_table.number.tolist()
symbol = periodic_table.symbol.tolist()

# Input substrate crystal
#substrate = Crystal.from_cif(input('***Input substrate cif file*** \n'))
substrate = Crystal.from_cif('cc.cif')

# Input overlayer crystal
#overlayer = Crystal.from_cif(input('\n***Input overlayer cif file*** \n'))
overlayer = Crystal.from_cif('cc.cif')

# rotation symmetry of substrate/overlayer
print("\nWood's notation: [ n sqrt(k) x n sqrt(k)) ] RR \n")
#n_woods = int(input('\n***Enter n: \n'))
#k_woods = int(input('\n***Enter k: \n'))
#R_woods = float(input('\n***Enter R: \n'))

n_o = 5
k_o = 1
R_o = 0

n_s = 2
k_s = 3 
R_s = 30

###################### EXTRACTING ATOMIC INFO FROM CIF FILES #######################
# lattice vectors
a_s, b_s, c_s = substrate.lattice_vectors
a_o, b_o, c_o = overlayer.lattice_vectors

# creating unit cells from lattice vectors
unitcell_substrate = np.array([a_s, b_s, c_s])
unitcell_overlayer = np.array([a_o, b_o, c_o])

# substrate expanded by sqrt(k_woods) x sqrt (k_woods) 
a_s = a_s * n_s * np.sqrt(k_s)
b_s = b_s * n_s * np.sqrt(k_s)
c_s = 3.0 * (c_s + c_o)  # adding vacuum to accomodate overlayer

# overlayer vectors extended
a_o = a_o * n_o * np.sqrt(k_o)
b_o = b_o * n_o * np.sqrt(k_o)
c_o = c_s

# creating unit cells from vectors
cell_vectors_substrate = np.array([a_s, b_s, c_s])
cell_vectors_overlayer = np.array([a_o, b_o, c_o])

# unit cell substrate atoms and symbols
symbols_substrate, atoms_substrate, num_atoms_substrate = extract_atoms_from_crystal(substrate, number_of_atoms=True)

# unit cell overlayer atoms and symbols
symbols_overlayer, atoms_overlayer, num_atoms_overlayer = extract_atoms_from_crystal(overlayer, number_of_atoms=True)

# estimating reasonable interlayer separation based on bond length
lengths = pdist(atoms_overlayer, 'euclidean')
bond_distance = min(lengths)
interlayer_distance =  10.0 * bond_distance

#################################### SELECT INITIAL ATOM ################################

initial_atoms_substrate =  np.dot(atoms_substrate,  np.linalg.inv(cell_vectors_substrate))
initial_atoms_overlayer =  np.dot(atoms_overlayer,  np.linalg.inv(cell_vectors_overlayer))
initial_atoms = list(atoms_substrate) + list(atoms_overlayer)

i, j = 0, 0
print("\nInitial SUBSTRATE atoms:")
for atom in initial_atoms_substrate:
    print('{} {:2} {:12.6f}  {:12.6f}  {:12.6f}'.format(i, symbols_substrate[i], atom[0], atom[1], atom[2]))
    i+=1

print("\nInitial OVERLAYER atoms:")
for atom in initial_atoms_overlayer:
    print('{} {:2} {:12.6f}  {:12.6f}  {:12.6f}'.format(i+j, symbols_overlayer[j], atom[0], atom[1], atom[2]))
    j+=1

idx_init_substrate = input('\nEnter Atom No. (Press Enter to skip and use default [0, 0, 0]):\n')
if idx_init_substrate.isdigit() and int(idx_init_substrate) >= 0 and int(idx_init_substrate) <= i:
    idx_init_substrate = int(idx_init_substrate)
    initial_atom = initial_atoms[idx_init_substrate]
    print("\nSelected origin: ", initial_atom)
else:
    initial_atom = np.array([0, 0, 0])
    print("Using default origin: ", initial_atom)


#initial_atoms_overlayer =  np.dot(atoms_overlayer,  np.linalg.inv(cell_vectors_overlayer))
#print("\nInitial Overlayer atoms..")
#for i, atom in enumerate(initial_atoms_substrate):
#    print('{} {:2} {:12.6f}  {:12.6f}  {:12.6f}'.format(i, symbols_overlayer[i], atom[0], atom[1], atom[2]))
#print("\n")


############## SHIFTING Z COORDINATES TO MAKE SURE EVERYTHING FITS WELL ###########

# lowest z atom in overlayer, i.e., atom closest to substrate
bottom_atom_overlayer = lowest(overlayer) # extracts only z coordinate

# pushing atoms to the bottom of the unit cell
bottom_atom_substrate = lowest(substrate) 
atoms_substrate = add_value_to_z_coordinates(atoms_substrate, -bottom_atom_substrate)

# highest atom in substrate after shifting the substrate
top_atom_substrate = highest(substrate) - bottom_atom_substrate

# positioning overlayer above substrate
z_start = top_atom_substrate + interlayer_distance
atoms_overlayer = add_value_to_z_coordinates(atoms_overlayer, -bottom_atom_overlayer)
atoms_overlayer = add_value_to_z_coordinates(atoms_overlayer, z_start)

############################ BUILDING SUPERCELLS ##################################

# supercell size subsratate
scaling_s = int(1.5 * np.sqrt(n_s))
supercell_size_substrate = [scaling_s*5, scaling_s*5, 1]

# supercell substrate
supercell_substrate_atoms, supercell_substrate_symbols = supercell_generator(atoms_substrate, symbols_substrate, unitcell_substrate,
        supercell_size_substrate)

#supercell size overlayer
scaling_o = int(1.5 * np.sqrt(n_o))
supercell_size_overlayer = [scaling_o*5, scaling_o*5, 1]

# supercell overlayer
supercell_overlayer_atoms, supercell_overlayer_symbols = supercell_generator(atoms_overlayer, symbols_overlayer, unitcell_overlayer,
        supercell_size_overlayer)

# Find central atoms in supercell
central_atom_substrate = find_most_central_atom(supercell_substrate_atoms)
central_atom_overlayer = find_most_central_atom(supercell_overlayer_atoms)

# origin 
origin_substrate = np.array([initial_atom[0], initial_atom[1], 0.0])
#print("Rotation axis ",origin_substrate)
#origin_substrate = np.array([0, 0, 0])
#origin_overlayer = np.array([0, 0, 0])
######################## ROTATION OPERATIONS ################################

##### SUBSTRATE ROTATION #####

# Let's define the rotation matrix
rotation_mat_substrate = rotation_matrix(R_s)

# rotated cell
rotated_vectors_substrate = np.dot(rotation_mat_substrate, cell_vectors_substrate.T).T
#rotated_vectors_substrate = cell_vectors_substrate
#rotated_vectors_substrate = rotation.apply(cell_vectors_substrate)

# rotating substrate atoms
supercell_substrate_atoms = supercell_substrate_atoms - origin_substrate
#supercell_substrate_atoms = np.dot(rotation_mat_substrate, supercell_substrate_atoms.T).T

# get vertices of the rotated cell
v1, v2, v3, v4 = vertices_unitcell(rotated_vectors_substrate[0],
        rotated_vectors_substrate[1], origin_substrate)

# vertices substrate
vertices_substrate = v1, v2, v3, v4
vertices_substrate = vertices_substrate

#### OVELAYER ROTATION ######

# Let's define the rotation matrix
#rotation_mat_overlayer = Rotation.from_euler('z', R_o, degrees=True).as_matrix()

# rotated cell
#rotated_vectors_overlayer = np.dot(rotation_mat_overlayer, cell_vectors_overlayer.T).T

# rotating overlayer atoms
supercell_overlayer_atoms = supercell_overlayer_atoms - origin_substrate
#supercell_overlayer_atoms = np.dot(rotation_mat_overlayer, supercell_overlayer_atoms.T).T

# get vertices of the rotated cell
#v1, v2, v3, v4 = vertices_unitcell(rotated_vectors_overlayer[0],
#        rotated_vectors_overlayer[1], origin_overlayer)

# vertices substrate
#vertices_overlayer = v1, v2, v3, v4



################### ATOMS IN POLYGON (UNIT CELL) ##############################

# rotate supercell atoms in substrate to align with rotated cell
#supercell_substrate_atoms = np.dot(supercell_substrate_atoms,
#        rotation_mat_substrate)

## substrate atoms present in rotated unit cell
substrate_atoms_in_rotated_cell, substrate_symbols_in_rotated_cell = atoms_in_polygon(vertices_substrate, 
    supercell_substrate_atoms, supercell_substrate_symbols)

## overlayer atoms present in rotated unit cell
overlayer_atoms_in_rotated_cell, overlayer_symbols_in_rotated_cell = atoms_in_polygon(vertices_substrate,
    supercell_overlayer_atoms, supercell_overlayer_symbols)


############################ COMBINING SELECTED ATOMS ###########################

# selected subsrate atoms
selected_atoms_substrate = list(substrate_atoms_in_rotated_cell)
selected_symbols_substrate =  substrate_symbols_in_rotated_cell

# selected overlayer atoms
selected_atoms_overlayer = list(overlayer_atoms_in_rotated_cell)
selected_symbols_overlayer =  overlayer_symbols_in_rotated_cell

# TOTAL selected atoms
selected_atoms = selected_atoms_overlayer
selected_symbols = selected_symbols_overlayer
#selected_atoms = selected_atoms_substrate
#selected_symbols = selected_symbols_substrate

#selected_atoms = selected_atoms_substrate + selected_atoms_overlayer
#selected_symbols = selected_symbols_substrate + selected_symbols_overlayer

######################## COVERTING TO FRACTIONALCOORDINATES #####################

# overlayer angnstrom to fractional
selected_atoms = cartesian_to_fractional(selected_atoms, rotated_vectors_substrate)

####################### REMOVING EQUIVALENT ATOMS ###############################

# start by adjusting all negative coordinate back to positive
#selected_atoms = adjust_coordinates(selected_atoms)

# removing equivalent atoms
#selected_atoms, selected_symbols = remove_similar_atoms(selected_atoms, selected_symbols)

#print("before transformation: ",selected_atoms, selected_symbols)
selected_atoms, selected_symbols = transform_atoms(selected_atoms, selected_symbols)
#print("selected_atoms", selected_atoms, selected_symbols)

########################## SORT ATOMS BY Z COODINATE #############################


# sort atoms wrt to z coordinate
selected_atoms, selected_symbols = sort_atoms_by_z_coordinates(selected_atoms,
        selected_symbols)

print(f"\nATOMIC_POSITIONS crystal")
for i in range(len(selected_atoms)):
    print('{:2}  {:12.6f}  {:12.6f}  {:12.6f}'.format(selected_symbols[i],
        selected_atoms[i][0], selected_atoms[i][1], selected_atoms[i][2]))

################################ UNIT CELL ######################################

print('\nCELL_PARAMETERS angstrom')
for vector in rotated_vectors_substrate:
    print('  {:12.6f}  {:12.6f}  {:12.6f}'.format(vector[0], vector[1], vector[2]))
 
######################## WRITING QUANTUM ESPRESSO FILE ##########################

# generate quantum espresso input file
qe_input_file = generate_qe_input_file(rotated_vectors_substrate,
                                        selected_atoms,
                                        selected_symbols)

with open('qe_input.in', 'w') as f:
    f.write(qe_input_file)

################################### PLOTTING ####################################

# plot atoms in rotated cell and show roated and unrotated vectors
#plot_cell(vertices_substrate, rotated_vectors_substrate, 
#        cell_vectors_substrate,
#        supercell_substrate_atoms,
#        supercell_overlayer_atoms)


############################## SUMMARY REPORT ####################################

print ("\n********************* SUMMARY REPORT ************************")
print("\nLattice constant of substrate: ", np.linalg.norm(rotated_vectors_substrate[0]))
#print("\nLattice constant of overlayer: ", np.linalg.norm(rotated_vectors_overlayer[0]))
print("\nTotal number of atoms =  ", len(selected_atoms))
print ('\n*************************** Done!! **************************\n')


