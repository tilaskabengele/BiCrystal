# This is a file with helper functions for program.py
# ***************************************************

# Imported packages
import math
import numpy as np
from shapely.geometry import Point, Polygon
from crystals import Element

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

def calculate_relative_rotation_angle(m, n):
    relative_rotation_angle = np.arccos((m**2 + 4*m*n + n**2) / (2 * (n**2 + m*n + m**2)))
    relative_rotation_angle_degrees = np.rad2deg(relative_rotation_angle)
    return relative_rotation_angle_degrees


################################################################################
#
#******************************** HETEROSTRUCTURES.PY **************************

#*********************************
# rotation matrix
# input: angle in degrees
# output: matrix
# ********************************
def rotation_matrix(angle_deg):
    # Convert angle from degrees to radians
    angle_rad = np.radians(angle_deg)
    
    # Compute the sine and cosine of the angle
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    
    # Create the rotation matrix
    rotation_mat = np.array([[cos_theta, -sin_theta, 0],
                            [sin_theta, cos_theta, 0],
                            [0, 0, 1]])
    
    return rotation_mat


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_cell_vectors(rotated_unit_cell, non_rotated_unit_cell):
    # Creating a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting lines for each vector in the rotated unit cell
    for vector in rotated_unit_cell:
        x, y, z = vector
        ax.plot([0, x], [0, y], [0, z], color='red')

    # Plotting lines for each vector in the non-rotated unit cell
    for vector in non_rotated_unit_cell:
        x, y, z = vector
        ax.plot([0, x], [0, y], [0, z], color='blue')

    # Setting plot limits and labels
#    ax.set_xlim([-5, 5])
#    ax.set_ylim([-5, 5])
#    ax.set_zlim([-5, 5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Displaying the plot
    plt.show()


def vertices_unitcell(a, b, origin):

        v1 = origin
        v2 = v1 + a
        v3 = v2 + b
        v4 = v3 - a

#        new_a = v3 - v2
#        new_b = v2 - v1

        return v1, v2, v3, v4

def extract_rotation_axis(a, b):
    cross_product = np.cross(a, b)
    axis = cross_product / np.linalg.norm(cross_product)
    return axis


def plot_polygon(vertices):
    # Extract the x, y, and z coordinates from the vertices
    x_coords = [v[0] for v in vertices]
    y_coords = [v[1] for v in vertices]
    z_coords = [v[2] for v in vertices]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Connect the vertices with lines
    for i in range(len(vertices)-1):
        ax.plot([x_coords[i], x_coords[i+1]], [y_coords[i], y_coords[i+1]], [z_coords[i], z_coords[i+1]], 'r-')

    # Connect the last vertex with the first vertex to complete the polygon
    ax.plot([x_coords[-1], x_coords[0]], [y_coords[-1], y_coords[0]], [z_coords[-1], z_coords[0]], 'r-')

    # Set the plot limits and labels
#    ax.set_xlim(min(x_coords), max(x_coords))
#    ax.set_ylim(min(y_coords), max(y_coords))
#    ax.set_zlim(min(z_coords), max(z_coords))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Display the plot
    plt.show()

def angle_between_vectors(v1, v2):
    # Convert the vectors to numpy arrays
    v1 = np.array(v1)
    v2 = np.array(v2)

    # Normalize the vectors
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    v1_normalized = v1 / v1_norm
    v2_normalized = v2 / v2_norm

    # Calculate the dot product
    dot_product = np.dot(v1_normalized, v2_normalized)

    # Calculate the angle in radians
    angle_radians = np.arccos(dot_product)

    # Convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

def extract_atoms_from_crystal(my_crystal, number_of_atoms=False):
    atoms = []
    symbols = []
    nat = 0 # keeping track of number of atoms
    for atm in my_crystal:
        nat+=1
        atoms.append(atm.coords_cartesian)
        symbols.append(str(atm))
    atoms = np.array(atoms)
    if number_of_atoms == True:
        return symbols, np.array(atoms), nat
    else:
        return symbols, np.array(atoms)

def supercell_generator(atoms, symbols, cell_vectors, supercell_size):
    replicated_atoms = []
    replicated_symbols = []

    num_atoms = len(atoms)
    
    # supercell in the positive axes
    # ******************************
    for k in range(supercell_size[2]):
        for j in range(supercell_size[1]):
            for i in range(supercell_size[0]):
                for atom_index in range(num_atoms):
                    symbol = symbols[atom_index]
                    replicated_atoms.append(atoms[atom_index] 
                            + i * cell_vectors[0] 
                            + j * cell_vectors[1] 
                            + k * cell_vectors[2])
                    replicated_symbols.append(symbol)
    
    # supercell in the negative axes
    # ******************************
    for k in range(supercell_size[2]):
        for j in range(supercell_size[1]):
            for i in range(supercell_size[0]):
                for atom_index in range(num_atoms):
                    symbol = symbols[atom_index]
                    replicated_atoms.append(atoms[atom_index]
                            - i * cell_vectors[0] 
                            - j * cell_vectors[1]
                            + k *  cell_vectors[2])
                    replicated_symbols.append(symbol)


    # supercell in the x-negative y-positive axes
    # *******************************************
    for k in range(supercell_size[2]):
        for j in range(supercell_size[1]):
            for i in range(supercell_size[0]):
                for atom_index in range(num_atoms):
                    symbol = symbols[atom_index]
                    replicated_atoms.append(atoms[atom_index]
                            - i * cell_vectors[0] 
                            + j * cell_vectors[1]
                            + k * cell_vectors[2])
                    replicated_symbols.append(symbol)


    # supercell in the y-negative x-positive axes
    # *******************************************
    for k in range(supercell_size[2]):
        for j in range(supercell_size[1]):
            for i in range(supercell_size[0]):
                for atom_index in range(num_atoms):
                    symbol = symbols[atom_index]
                    replicated_atoms.append(atoms[atom_index]
                            + i * cell_vectors[0] 
                            - j * cell_vectors[1]
                            + k * cell_vectors[2])
                    replicated_symbols.append(symbol)


    supercell_atoms = np.array(replicated_atoms)
    supercell_symbols = np.array(replicated_symbols)

    return supercell_atoms, supercell_symbols

from shapely.geometry import Point, Polygon, LineString

def atoms_in_polygon(vertices, atomic_positions, symbols):
    polygon = Polygon(vertices)
    edge = LineString(vertices)
    atoms_within_polygon = []
    symbols_within_polygon = []

    for position, symbol in zip(atomic_positions, symbols):
        point = Point(position)
        if polygon.contains(point) or edge.distance(point) == 0:
            atoms_within_polygon.append(position)
            symbols_within_polygon.append(symbol)

    return np.array(atoms_within_polygon), symbols_within_polygon

def generate_qe_input_file(cell_vectors, atomic_positions, symbols):
    input_file = ''

    input_file += '&control\n'
    input_file += '    calculation = "scf"\n'
    input_file += '    prefix = "qe_structure"\n'
    input_file += '    pseudo_dir = "./pseudopotentials/"\n'
    input_file += '    outdir = "./output/"\n'
    input_file += '/\n\n'

    input_file += '&system\n'
    input_file += '    ibrav = 0\n'
    input_file += '    nat = {}\n'.format(len(atomic_positions))
    input_file += '    ntyp = {}\n'.format(len(set(symbols)))
    input_file += '    ecutwfc = 50\n'
    input_file += '    ecutrho = 200\n'
    input_file += '/\n\n'

    input_file += '&electrons\n'
    input_file += '    diagonalization = "david"\n'
    input_file += '    conv_thr = 1.0d-8\n'
    input_file += '    mixing_beta = 0.7\n'
    input_file += '/\n\n'

    input_file += '&ions\n/\n\n'

    input_file += '&cell\n'
    input_file += '/\n\n'

    input_file += 'ATOMIC_SPECIES\n'
    unique_symbols = list(set(symbols))
    for sym in unique_symbols:
        input_file += '{}   {}   {}.UPF\n'.format(sym, Element(sym).mass, sym.lower())

    input_file += '\n'

    input_file += 'ATOMIC_POSITIONS crystal\n'
    for i in range(len(atomic_positions)):
        input_file += '{:2}  {:12.6f}  {:12.6f}  {:12.6f}\n'.format(symbols[i], atomic_positions[i][0], atomic_positions[i][1], atomic_positions[i][2])

    input_file += '\nK_POINTS automatic\n'
    input_file += '2 2 2 0 0 0\n\n'

    input_file += 'CELL_PARAMETERS angstrom\n'
    for vector in cell_vectors:
        input_file += '  {:12.6f}  {:12.6f}  {:12.6f}\n'.format(vector[0], vector[1], vector[2])
 

    return input_file

#def remove_duplicate_atoms(coords, symbols):
#    unique_indices = np.unique(coords, axis=0, return_index=True)[1]
#    unique_coords = coords[unique_indices]
#    unique_symbols = [symbols[i] for i in unique_indices]
#    
#    return unique_coords, unique_symbols

def sort_atoms_by_z_coordinates(atoms, symbols):
    if len(atoms) <= 0:
        print("\nWarning: Empty array in def sort_atoms_by_z_coordinates\n")
        return atoms, symbols
    z_coordinates = atoms[:, 2]  # Extract z coordinates from the atoms array
    sorted_indices = np.argsort(z_coordinates)[::-1]  # Sort the indices in descending order based on z coordinates

    sorted_atoms = atoms[sorted_indices]  # Rearrange the atoms array based on sorted indices
    sorted_symbols = [symbols[i] for i in sorted_indices]  # Rearrange the symbols list based on sorted indices

    return sorted_atoms, sorted_symbols

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def calculate_length(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def calculate_angle(point1, point2, point3):
    vector1 = np.array(point2) - np.array(point1)
    vector2 = np.array(point3) - np.array(point2)
    dot_product = np.dot(vector1, vector2)
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    return np.arccos(dot_product / norm_product)

def plot_cell(vertices, rotated_unit_cell, non_rotated_unit_cell, atoms1, atoms2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a list of the polygon vertices
    polygon_vertices = [list(vertex) for vertex in vertices]

    # Add the polygon to the plot
    polygon = Poly3DCollection([polygon_vertices])
    polygon.set_alpha(0.5)
    polygon.set_facecolor('blue')
    ax.add_collection3d(polygon)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
#    plt.title('Polygon')

    # Calculate and print the length of each side
    num_vertices = len(vertices)
    for i in range(num_vertices):
        j = (i + 1) % num_vertices
        length = calculate_length(vertices[i], vertices[j])
        #print(f"Length of side {i+1}: {length}")

    # Calculate and print the angle between each pair of sides
    for i in range(num_vertices):
        j = (i + 1) % num_vertices
        k = (i + 2) % num_vertices
        angle = calculate_angle(vertices[i], vertices[j], vertices[k])
        #print(f"Angle between side {i+1} and side {i+2}: {np.degrees(angle)} degrees")

    #*************************** cell vectors ****************
    # Plotting lines for each vector in the rotated unit cell
    for vector in rotated_unit_cell:
        x, y, z = vector
        ax.plot([0, x], [0, y], [0, z], color='red')

    # Plotting lines for each vector in the non-rotated unit cell
    for vector in non_rotated_unit_cell:
        x, y, z = vector
        ax.plot([0, x], [0, y], [0, z], color='blue')
    
    d = int(round(len(atoms1)/8))
    # Setting plot limits and labels
#    ax.set_xlim([-d, d])
#    ax.set_ylim([-d, d])
#    ax.set_zlim([-d, d])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    #********************** add atoms to cell ***************
    ax.scatter(atoms1[:, 0], atoms1[:, 1], atoms1[:, 2], s=5, c='tab:red', alpha=0.4)
    ax.scatter(atoms2[:, 0], atoms2[:, 1], atoms2[:, 2], s=5, c='tab:green', alpha=0.4)

    # Displaying the plot
    plt.show()

def find_most_central_atom(coordinates):
    centroid = np.mean(coordinates, axis=0)
    distances = np.linalg.norm(coordinates - centroid, axis=1)
    central_atom_index = np.argmin(distances)
    central_atom = coordinates[central_atom_index]
    return central_atom

def add_value_to_z_coordinates(atom_list, v):
    atom_array = np.array(atom_list)
    atom_array[:, 2] += v
    return atom_array

def cartesian_to_fractional(atoms_cartesian, cell_vectors):
    atoms_cartesian = np.array(atoms_cartesian)
    transformation_matrix = np.linalg.inv(cell_vectors)
    atoms_fractional = np.dot(atoms_cartesian, transformation_matrix)
    return atoms_fractional

def remove_similar_atoms(atoms, symbols):
    filtered_atoms = []
    filtered_symbols = []
    
    i = -1
    for atom in atoms:
        i+=1
        if atom[0] < 1 and atom[1] < 1 and atom[2] < 1:
            filtered_atoms.append(np.array(atoms[i]))
            filtered_symbols.append(symbols[i])
    return np.array(filtered_atoms), filtered_symbols

def adjust_coordinates(coords):
    adjusted_coords = coords.copy()
    for i, atom in enumerate(coords):
        if np.round(atom[0], 6) < 0:
            adjusted_coords[i][0] = atom[0] + 1
        if np.round(atom[1], 6) < 0:
            adjusted_coords[i][1] = atom[1] + 1
        if np.round(atom[2], 6) < 0:
            adjusted_coords[i][2] = atom[2] + 1
    return adjusted_coords

"""
from collections import defaultdict
def transform_atoms(atoms, symbols):
    transformed_atoms = []
    transformed_symbols = []
    unique_atoms = set()

    for atom, symbol in zip(atoms, symbols):
        x, y, z = atom

        # Transformation 1: Add 1 to x or y coordinates less than 0
        if x < 0:
            x += 1
        if y < 0:
            y += 1

        # Transformation 5: Convert values close to 1 to zeros
        if np.allclose(x, 1):
            x = 0
        if np.allclose(y, 1):
            y = 0

        # Transformation 3: Treat values close to zero as zero
        if np.allclose(x, 0):
            x = 0
        if np.allclose(y, 0):
            y = 0
        if np.allclose(z, 0):
            z = 0

        # Transformation 2: Remove duplicates based on symbol, x, y, and z coordinates
        coord = (x, y, z, symbol)
        duplicate = False
        for unique_coord in unique_atoms:
            if np.allclose(coord[:3], unique_coord[:3], atol=1e-2) and coord[3] == unique_coord[3]:
                duplicate = True
                break

        if duplicate:
            continue

        unique_atoms.add(coord)
        transformed_atoms.append([x, y, z])
        transformed_symbols.append(symbol)

    final_atoms = np.array(transformed_atoms)

    # Transformation 6: Remove duplicate atoms and corresponding symbols
    unique_atoms.clear()
    unique_transformed_atoms = []
    unique_transformed_symbols = []
    for atom, symbol in zip(transformed_atoms, transformed_symbols):
        coord = (tuple(atom), symbol)
        if coord in unique_atoms:
            continue

        unique_atoms.add(coord)
        unique_transformed_atoms.append(atom)
        unique_transformed_symbols.append(symbol)

    final_atoms = np.array(unique_transformed_atoms)
    final_symbols = unique_transformed_symbols

    return final_atoms, final_symbols
    """


def transform_atoms(coordinates, symbols):
    # Transformation 1: Scale the x and y coordinates by a factor of 1
    scaled_coordinates = coordinates.copy()
    scaled_coordinates[:, :2] *= 1

    # Transformation 2: Apply periodic boundary conditions to x and y coordinates
    wrapped_coordinates = np.mod(scaled_coordinates[:, :2], 1)
    wrapped_coordinates = np.concatenate((wrapped_coordinates, scaled_coordinates[:, 2:]), axis=1)

    # Transformation 3: Remove atoms with fractional coordinates outside the unit cell
    inside_unit_cell = np.logical_and(wrapped_coordinates[:, :2] >= 0, wrapped_coordinates[:, :2] <= 1)
    filtered_coordinates = wrapped_coordinates[np.all(inside_unit_cell, axis=1)]
    filtered_symbols = np.array(symbols)[np.all(inside_unit_cell, axis=1)]

    # Transformation 4: Remove duplicate atoms and corresponding symbols
    unique_coordinates, unique_indices = np.unique(filtered_coordinates, axis=0, return_index=True)
    unique_symbols = filtered_symbols[unique_indices]

    # Transformation 5: Remove atoms greater than 1 and equivalent atoms up to a factor of 1
    unique_indices = []
    for i, (coord, symbol) in enumerate(zip(unique_coordinates, unique_symbols)):
        # Check if the atom is greater than 1 in x, y, or z coordinates
        if np.any(coord > 1):
            continue

        # Check if the atom is equivalent up to a factor of 1 in x, y, and z coordinates
        equivalent = False
        for j, (unique_coord, unique_symbol) in enumerate(zip(unique_coordinates[:i], unique_symbols[:i])):
            if np.allclose(coord, unique_coord, atol=1e-9) and symbol == unique_symbol:
                equivalent = True
                break

        if not equivalent:
            unique_indices.append(i)

    unique_coordinates = unique_coordinates[unique_indices]
    unique_symbols = unique_symbols[unique_indices]

    # Transformation 6: Convert trailing negative signs in coordinates to 0.000000 if the absolute value is less than a threshold
    abs_coords = np.abs(unique_coordinates)
    unique_coordinates = np.where(abs_coords < 1e-8, 0.0, unique_coordinates)

    return unique_coordinates, unique_symbols

