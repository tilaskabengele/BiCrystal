# This is a file with helper functions for program.py
# ***************************************************

# Imported packages
import math
import numpy as np
from shapely.geometry import Point, Polygon

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


