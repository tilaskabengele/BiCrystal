#This subroutine plots the results from program.py
# **************************************************

# Importing required packages...
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import path
from sklearn.neighbors import KNeighborsRegressor
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors
import json

# helper functions
from helpers_bicrystal import *

#**********************************************************
# Read variables from file
with open('atomic_plotting_data.json', 'r') as file:
    data = json.load(file)
    v1rb = np.array(data['v1rb'])
    v2rb = np.array(data['v2rb'])
    v3rb = np.array(data['v3rb'])
    v4rb = np.array(data['v4rb'])
    v1rt = np.array(data['v1rt'])
    v2rt = np.array(data['v2rt'])
    v3rt = np.array(data['v3rt'])
    v4rt = np.array(data['v4rt'])
    atm = np.array(data['atm'])
    zb = np.array(data['zb'])
    zt = np.array(data['zt'])
    bond_distance = data['bond_distance']
    bot = np.array(data['bot'])
    symb_num_bot = data['symb_num_bot']
    symb_num_top = data['symb_num_top']
    botl = data['botl']
    top = np.array(data['top'])
    topl = data['topl']
    coord_b = data['coord_b']
    coord_t = data['coord_t']

#**********************************************
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

# UNCOMMENT FOR DARK BACKGROUND
# ******************************
#ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
#ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
#ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

# extracting x and y coordinates
xb, yb = zip(*coord_b)

# plotting boundary at lowest z position
plt.plot(xb,yb,zb)

# plotting bottom layer atoms
bot = bot.T
supx,supy,supz = list(bot)
ax.scatter(supx, supy, supz, s=10, c='tab:green', alpha=0.4)
ax.scatter(supx, supy, supz, c='tab:blue')

# polygon boundaries
boundary_bot,_,_,_,_ = poly(v1rb,v2rb,v3rb,v4rb)
boundary_top,_,_,_,_ = poly(v1rt,v2rt,v3rt,v4rt)

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

# extracting x and y coordinates
xt, yt = zip(*coord_t)

# plotting the boundary at hiest z position
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
