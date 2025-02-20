from heapq import heappush, heappop  # Recommended.
import numpy as np

from flightsim.world import World

from .occupancy_map import OccupancyMap # Recommended.

# def find_neighbors(center_index, g):
#     neighbors = []
#     for di in [-1, 0, 1]:
#         for dj in [-1, 0, 1]:
#             for dk in [-1, 0, 1]:
#                 if di == 0 and dj == 0 and dk == 0:
#                     continue
#                 else:
#                     neighbor_u = center_index + np.array([di, dj, dk])
#                     if np.any(neighbor_u) == 0:
#                         continue
#                     else:
#                         neighbor_g_u = g[neighbor_u]
#                         neighbors.append((neighbor_g_u, neighbor_u))
#     return neighbors

def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        return a tuple (path, nodes_expanded)
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
        nodes_expanded, the number of nodes that have been expanded
    """

    # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))
    # cube_shape = (3, 3, 3)

    g = {start_index: 0} # distances
    parents = {} # parents
    # g = float('inf') * np.ones(cube_shape)  # cost
    # p = False * np.ones(cube_shape)  # parents

    # Q
    # Q = find_neighbors(start_index)  # no path found yet
    Q = [(0, start_index)] # priority queue = (cost, id)

    g[start_index] = 0

    nodes_expanded = []

    goal_voxel_center = occ_map.index_to_metric_center(goal_index)

    # if astar:
    #     # A*
    #     while goal_index in Q:
    #         g_u, u = heappop(Q)
    #         Q.remove(u)
    #
    #
    # else:


    while Q:
        (distance, current) = heappop(Q)
        nodes_expanded.append(current)
        if distance > g[current]:
            continue
        neighbors = np.array([(-1,-1,-1), (-1,-1,0), (-1,-1,1),
                              (-1,0,-1), (-1,0,0), (-1,0,1),
                              (-1,1,-1), (-1,1,0), (-1,1,1),
                              (0,-1,-1), (0,-1,0), (0,-1,1),
                              (0,0,-1), (0,0,1),
                              (0,1,-1), (0,1,0), (0,1,1),
                              (1,-1,-1), (1,-1,0), (1,-1,1),
                              (1,0,-1), (1,0,0), (1,0,1),
                              (1,1,-1), (1,1,0), (1,1,1)])

        for n in neighbors:
            new_voxel_center = occ_map.index_to_metric_center(current + n)
            new_voxel_id = tuple(occ_map.metric_to_index(new_voxel_center))
            if not occ_map.is_valid_index(new_voxel_id) or occ_map.is_occupied_index(new_voxel_id):
                continue

            current_voxel_center = occ_map.index_to_metric_center(current)
            new_voxel_center = occ_map.index_to_metric_center(new_voxel_id)
            new_distance = g[current] + np.linalg.norm(new_voxel_center - current_voxel_center, ord=2)

            if astar:
                # A*
                new_distance = new_distance + np.linalg.norm(new_voxel_center - goal_voxel_center, ord=2)

            # if astar:
            #     # A*
            #     new_distance = g[current] + np.linalg.norm(current_voxel_center - goal_voxel_center)
            #
            # else:
            #     # Dijkstras
            #     new_distance = g[current] + np.linalg.norm(new_voxel_center - current_voxel_center, ord=2)

            if (new_voxel_id not in g) or (new_distance < g[new_voxel_id]):
                g[new_voxel_id] = new_distance
                parents[new_voxel_id] = current

                heappush(Q, (new_distance, new_voxel_id))

        # while goal_index in Q:
        #     g_u, u = heappop(Q)
        #     Q.remove(u)
        #     nodes_expanded.append(u)
        #
        #     for g_v, v in Q:
        #         if v in find_neighbors(u):
        #             cost_u_v = np.linalg.norm(u - v, ord=2)
        #             d = g[u] + cost_u_v
        #             if d < g[v]:
        #                 g[v] = d
        #                 p[v] = u
        #                 heappush(Q, (g[v], v))

    # Return a tuple (path, nodes_expanded)
    # path
    path = []
    voxel_id = goal_index
    while voxel_id is not None and voxel_id != start_index:
        path.append(occ_map.index_to_metric_negative_corner(voxel_id))
        voxel_id = parents[voxel_id]
    path.reverse()
    path[0] = np.array(start)
    path[-1] = np.array(goal)
    path = np.array(path)
    # path = []
    # while v is not None:
    #     path.append(v)
    #     v = p[v]
    # path.reverse()

    return_tuple = (path, nodes_expanded)

    return return_tuple
