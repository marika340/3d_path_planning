import inspect
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time

from meam620.flightsim.axes3ds import Axes3Ds
from meam620.flightsim.world import World

# from proj1_2.code.occupancy_map import OccupancyMap
from meam620.proj1_2.code.graph_search import graph_search

# Choose a test example file. You should write your own example files too!
# filename = 'test_empty.json'
filename = 'test_slide.json'

# Load the test example.
file = Path(inspect.getsourcefile(lambda:0)).parent.resolve() / '..' / 'util' / filename
world = World.from_file(file)          # World boundary and obstacles.
# resolution = world.world['resolution'] # (x,y,z) resolution of discretization, shape=(3,).
margin = world.world['margin']         # Scalar spherical robot size or safety margin.
start  = world.world['start']          # Start point, shape=(3,)
goal   = world.world['goal']           # Goal point, shape=(3,)

# Run your code and return the path.
# start_time = time.time()
resolutions = [(0.6, 0.6, 0.6), (0.9, 0.9, 0.9), (1.2, 1.2, 1.2), (1.5, 1.5, 1.5)]
x_labels = ["[0.6,0.6,0.6]", "[0.9,0.9,0.9]", "[1.2,1.2,1.2]", "[1.5,1.5,1.5]"]
nodes_expanded_dijkstra = []
nodes_expanded_astar = []
for resolution in resolutions:
    path, node_expanded = graph_search(world, resolution, margin, start, goal, astar=False)
    nodes_expanded_dijkstra.append(node_expanded)
for resolution in resolutions:
    path, node_expanded = graph_search(world, resolution, margin, start, goal, astar=True)
    nodes_expanded_astar.append(node_expanded)
# end_time = time.time()

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(x_labels, nodes_expanded_dijkstra, marker='o', linestyle='-', label="Dijkstra")
plt.plot(x_labels, nodes_expanded_astar, marker='s', linestyle='-', label="A*")

plt.xlabel("Resolution")
plt.ylabel("Nodes Expanded")
plt.title("Nodes Expanded vs Resolution for Dijkstra and A*")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Print results.
# print()
# print('Total number of nodes expanded:', node_expanded)
# print(f'Solved in {end_time-start_time:.2f} seconds')
# if path is not None:
#     number_points = path.shape[0]
#     length = np.sum(np.linalg.norm(np.diff(path, axis=0),axis=1))
#     print(f'The discrete path has {number_points} points.')
#     print(f'The path length is {length:.2f} meters.')
# else:
#     print('No path found.')
#
# # Draw the world, start, and goal.
# fig = plt.figure()
# ax = Axes3Ds(fig)
# world.draw(ax)
# ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=10, markeredgewidth=3, markerfacecolor='none')
# ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=10, markeredgewidth=3, markerfacecolor='none')
#
# # Plot your path on the true World.
# if path is not None:
#     world.draw_line(ax, path, color='blue')
#     world.draw_points(ax, path, color='blue')
#
# # For debugging, you can visualize the path on the provided occupancy map.
# # Very, very slow! Comment this out if you're not using it.
# # fig = plt.figure()
# # ax = Axes3Ds(fig)
# # oc = OccupancyMap(world, resolution, margin)
# # oc.draw(ax)
# # ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=10, markeredgewidth=3, markerfacecolor='none')
# # ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=10, markeredgewidth=3, markerfacecolor='none')
# # if path is not None:
# #     world.draw_line(ax, path, color='blue')
# #     world.draw_points(ax, path, color='blue')
#
# plt.show()
