# BiCrystal
``BiCrystal`` is a Python program that builds commensurate and incommensurate crystal structures of layered materials. The current version reads CIF files and writes the new structure to a QUANTUM ESPRESSO input file. The program also provides additional information such as the bond distance between atoms, lattice vectors in Bohr and Angstrom, and a simple 3D plot of each layer.

Contents
==========
**Overview** \
**Download** \
**Packages** \
**Files** \
**Installation** \
**Usage** \
**Examples** \
**References** \
**License** 

# Overview
Building unit cells of arbitrary size is often an inevitable task when studying the physical and mechanical properties of layered materials such as graphene, hexagonal Boron Nitride, transition metal dichalcogenides. Although most visualzation software such as Xcrysden, VESTA or Avogadro provide very powerful tools for analysing and manipulating periodic crystal structures, constructing large unit cells in bilayers with one of the layers perturbed can be very daunting. ``BiCrystal`` provides a convinient and easy way of creating new crystal structures of arbitrary size from CIF files.

# Download
The latest version of ``BiCrystal`` can be found on github:

https://github.com/tilaskabengele/BiCrystal/


**Contact**: Tilas Kabengele tilas.kabengele@dal.ca

# Packages
`BiCrystal` is a python-based program that uses Scipy and Shapely libraries. Additionally, the Crystal package, which is not part of the standard Python packages, should be installed. i.e Install via pip or conda:

     pip install crystals
or

    conda install -c conda-forge crystals
and

    pip install shapely
    
For more information on crystals and shapely, visit: https://pypi.org/project/crystals/ and https://pypi.org/project/Shapely/ respectively.
 
 # Files

**bicrystal** - Bash script which runs the python program \
**cifs/** - Directory with sample cif files \
**examples/** - Directory with 33 examples of QUANTUM ESPRESSO input files generated by BiCrystal \
**periodic_table.csv** - Periodic table of elements \
**program.py** - Python program to be called from bicrystal script

# Installation
After downloading the files from the github repository to the directory of your choice (_recommended: /usr/bin/_), make `bicrystal` and `program.py` into executables:

    chmod u+x bicrystal program.py

Next, add this directory to your $PATH variable. In Bash, adding the following lines to your `.bashrc file`:

    vi ~/.bashrc

Add:

    export PATH="$/path/to/your/directory/with/bicrysal/:$PATH"
    export PYTHONPATH="${PYTHONPATH}:/path/to/your/directory/with/bicrysal/"

Save, close then source your `.bashrc` file to activate the changes:
    
    source ~/.bashrc

Restart your terminal window to start using `bicrystal`.

# Usage
BiCrystal is an interactive program that instructs the user every step of the way. To start BiCrystal, in the terminal type:

    bicrystal
    
The first thing you will be required to do is input your cif file, e.g. graphite.cif:

    ***Input cif file***
    graphite.cif

Next, enter input parameters m and n, and rotation angle in degrees (_zero if you want both layers unperturbed_).
Parameter m and n correspond to the scale of the lattice vectors along the x and y directions, respectively. As an example, let's take m = 2, n = 1 and rotation angle 21.79 degrees.
    
    ***Rotation parameters***
    Enter m 2
    Enter n 1
    Enter rotation_angle 0

After that, you will be required to pick a zeroeth atom from the top and bottom layer. If we were picking the atoms by hand using a visualization software such as Xcrysden, this would be the atom we start from when creating the new cell vectors. 

    Intializing atoms...


    Initial TOP atoms..
    Atom No. 1   c   [0.  0.  0.5]
    Atom No. 2   c   [0.66667 0.33334 0.505  ]

    Initial BOTTOM atoms..
    Atom No. 3   c   [0. 0. 0.]
    Atom No. 4   c   [0.33333 0.66667 0.005  ]
    
 For a good symmetrical structure, always pick atoms such that the zeroeth TOP and BOTTOM atoms align. In this example, that would be Atom No. 1 and Atom No. 3. After picking your zeroeth atoms, a window with a simple 3D will then open.
 
    Select zeroeth TOP atom
    Enter Atom No. 1

    Select zeroeth BOTTOM atom
    Enter Atom No. 3

 
 ![cc28](https://user-images.githubusercontent.com/62076249/87927879-48795500-ca5a-11ea-98c1-b2949bb672e3.PNG)
 
 Finally, you can save your output as a QUANTUM ESPRESSO file and visualize with Xcrysden for a more sophisticated look.
 
     ********************* SUMMARY REPORT ***********************

    Top atoms(rotated) =  14
    Bottom atoms  =  14

    Total atoms
    = 28

    *************************** Done! **************************

    Would you like to write Espresso file?[Y/n]

   # Examples
   
   Let's say we saved our output in the example given above as graphite28.scf.in, we can visualize this with Xcrysden.
   
       xcrysden --pwi graphite28.scf.in
       
   ![moire28](https://github.com/tilaskabengele/BiCrystal/issues/2#issue-661672229)
   
   
   
   
   
