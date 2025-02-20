from heapq import heappush, heappop  # Recommended.
import numpy as np
import scipy.spatial.distance

from flightsim.world import World

from .occupancy_map import OccupancyMap # Recommended.

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

    g = {start_index: 0} # distances
    parents = {start_index: None} # parents

    # Q
    Q = [(0, start_index)] # priority queue = (cost, id)

    # g[start_index] = 0

    # nodes_expanded = []
    nodes_expanded = 0

    goal_voxel_center = occ_map.index_to_metric_center(goal_index)

    neighbors = np.array([(-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
                          (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
                          (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
                          (0, -1, -1), (0, -1, 0), (0, -1, 1),
                          (0, 0, -1), (0, 0, 1),
                          (0, 1, -1), (0, 1, 0), (0, 1, 1),
                          (1, -1, -1), (1, -1, 0), (1, -1, 1),
                          (1, 0, -1), (1, 0, 0), (1, 0, 1),
                          (1, 1, -1), (1, 1, 0), (1, 1, 1)])

    while Q:
        (distance, current) = heappop(Q)
        # nodes_expanded.append(current)
        nodes_expanded += 1

        if current == goal_index:
            break

        # if distance > g[current]:
        #     continue

        for n in neighbors:
            new_voxel_center = occ_map.index_to_metric_center(current + n)
            new_voxel_id = tuple(current + n)
            # new_voxel_id = tuple(occ_map.metric_to_index(new_voxel_center))
            if not occ_map.is_valid_index(new_voxel_id) or occ_map.is_occupied_index(new_voxel_id) or new_voxel_id in g:
                continue
                # pass
                # (new_voxel_id not in g)

            current_voxel_center = occ_map.index_to_metric_center(current)
            new_distance = g[current] + np.linalg.norm(np.array(new_voxel_center) - np.array(current_voxel_center))
            # new_distance = g[current] + scipy.spatial.distance.chebyshev(new_voxel_center, current_voxel_center)

            if astar:
                # A*
                h = np.linalg.norm(np.array(new_voxel_center) - np.array(goal_voxel_center))
                # new_distance = new_distance + h
                # h = scipy.spatial.distance.chebyshev(new_voxel_center, goal_voxel_center)
            else:
                h = 0.0
            #
            # new_distance = new_distance + h

            print('new_voxel_id', new_voxel_id)
            print('new_distance', new_distance)
            # print('g[new_voxel_id]', g[new_voxel_id])
            if (new_voxel_id not in g) or (new_distance < g[new_voxel_id]):
                g[new_voxel_id] = new_distance
                parents[new_voxel_id] = current

                heappush(Q, (new_distance + h, new_voxel_id))

    # Return a tuple (path, nodes_expanded)
    # path
    path = []
    voxel_id = goal_index
    if goal_index not in parents:
        return (None, nodes_expanded)
    while voxel_id is not None and voxel_id != start_index:
        path.append(occ_map.index_to_metric_center(voxel_id))
        voxel_id = parents[voxel_id]
    path.reverse()
    path[0] = np.array(start)
    path[-1] = np.array(goal)
    path = np.array(path)

    print(nodes_expanded)

    return (path, nodes_expanded)
