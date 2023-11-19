import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.neighbors import NearestNeighbors
import json

# Helper functions
from helpers_bicrystal import *

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

# Figure settings
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('white')
ax.grid(False)
ax.axis('off')

# Plotting boundary at lowest z position
ax.plot(*zip(*coord_b), zb, color='black')

# Plotting bottom layer atoms as points
ax.scatter(bot[:, 0], bot[:, 1], bot[:, 2], s=2, c='tab:green', alpha=0.4)

# Calculate boundary points for the bottom layer
boundary_bot, _, _, _, _ = poly(v1rb, v2rb, v3rb, v4rb)
boundary_top, _, _, _, _ = poly(v1rt, v2rt, v3rt, v4rt)

# Adding bonds to bottom layer
neigh = NearestNeighbors(radius=bond_distance + 0.1).fit(botl)
for i, atom in enumerate(botl):
    rng = neigh.radius_neighbors([list(atom)])
    dis = np.asarray(rng[0][0])
    idx = np.asarray(rng[1][0])
    if inpoly(atom, boundary_bot):
        for j in idx:
            atz = [atom, list(botl[j])]
            x_bond, y_bond, z_bond = zip(*atz)
            ax.plot(x_bond, y_bond, z_bond, color='black', linewidth=0.5)

# Plotting the boundary at highest z position
ax.plot(*zip(*coord_t), zt, color='black')

# Plotting top layer atoms as points
ax.scatter(top[:, 0], top[:, 1], top[:, 2], s=2, c='tab:red', alpha=0.4)

# Adding bonds to top layer
neigh = NearestNeighbors(radius=bond_distance + 0.1).fit(topl)
for i, atom in enumerate(topl):
    rng = neigh.radius_neighbors([list(atom)])
    dis = np.asarray(rng[0][0])
    idx = np.asarray(rng[1][0])
    if inpoly(atom, boundary_top):
        for j in idx:
            atz = [atom, list(topl[j])]
            x_bond, y_bond, z_bond = zip(*atz)
            ax.plot(x_bond, y_bond, z_bond, color='black', linewidth=0.5)

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Lock the views
ax.view_init(elev=90, azim=90)  # Top view
# ax.view_init(elev=0, azim=0)  # Side view
# ax.view_init(elev=0, azim=90)  # End view

plt.show()
