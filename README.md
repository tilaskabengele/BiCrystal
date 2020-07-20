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
**Summary Table** \
**References** \
**License** 

# Overview
Building unit cells of arbitrary size is often an inevitable task when studying the physical and mechanical properties of layered materials such as graphene, hexagonal Boron Nitride, transition metal dichalcogenides. Although most visualzation software such as Xcrysden, VESTA or Avogadro provide very powerful tools for analysing and manipulating periodic crystal structures, constructing large unit cells in bilayers with one of the layers perturbed can be very daunting. ``BiCrystal`` provides a convinient and easy way of creating new crystal structures of arbitrary size from CIF files.

# Download
The latest version of ``BiCrystal`` can be found on github:

https://github.com/tilaskabengele/BiCrystal/tree/stable

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
       
  ![ccmoire28](https://user-images.githubusercontent.com/62076249/87929694-377e1300-ca5d-11ea-80e3-76417f34a5e4.PNG)
   
Looking from the top view, we can see that for this rotation, a Moire pattern was created. Really neat! This was not apparent from the simple 3D plot because BiCrystal plots cartesian coordinates of the atoms where the top and bottom layer do not necessarily align. Before writing the QUANTUM ESPRESSO file, `BiCrystal` removes symmetrically equivalent atoms and maps back those atoms that fell outside the unit cell due to rotation. 

The **examples/** folder has over 30 examples of Moire patterns graphite, Molebdenum Disulfide and blue Phosphorene generated from `bicrystal`. Below are some examples.

# Graphite 364-atom unit cell
The unit cell of graphite with 364 atoms can be generated by using parameters: m = 6, n = 5, and rotation angle of 6.01 degrees. Shown below is the top view.

![cc364](https://user-images.githubusercontent.com/62076249/87933970-98f5b000-ca64-11ea-906b-15a1036989a1.PNG)

# Blue phosphorene 172-atom unit cell
The unit cell of blue phosphorene with 172 atoms can be generated by using parameters: m = 6, n = 1, and rotation angle of 44.82 degrees. Shown below is the top view.

![bluep172](https://user-images.githubusercontent.com/62076249/87934152-e245ff80-ca64-11ea-8f36-b69b5799e3fa.PNG)

# Molybdenun Disulfide 546-atom unit cell
The unit cell of MoS<sub>2</sub> with 546 atoms atoms can be generated by using parameters: m = 6, n = 5, and rotation angle of 6.01 degrees. Shown below is the top view.

![mos546](https://user-images.githubusercontent.com/62076249/87934312-2fc26c80-ca65-11ea-97a2-c9cdb3068a9d.PNG)

# Summary Table
All the examples in the examples folder can be summarized in the table below:

![example_table](https://user-images.githubusercontent.com/62076249/87934662-dc045300-ca65-11ea-8f54-818b7183d6e1.PNG)
   
 # References
 For a detailed analysis of Moire patterns and angles:
 
 **Density functional calculations on the intricacies of Moiré patterns on graphite**, J. M. Campanera, G. Savini, I. Suarez-Martinez, and M. I. Heggie, _Phys. Rev. B 75, 235449 – Published 28 June 2007_
   
 Crystals package authors:
 
 L. P. René de Cotret, M. R. Otto, M. J. Stern. and B. J. Siwick, An open-source software ecosystem for the interactive exploration of ultrafast electron scattering data, Advanced Structural and Chemical Imaging 4:11 (2018) DOI:10.1186/s40679-018-0060-y.
 
 For further reading and related projects, visit **Johnson Group wiki**: http://schooner.chem.dal.ca/wiki/Johnson_Group_Wiki 
 
 # License
 Copyright (c) 2020 Tilas Kabengele, Johnson Chemistry Group, Dalhousie University.

BiCrystal is a free program: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

BiCrystal is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
