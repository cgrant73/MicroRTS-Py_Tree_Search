import random
import numpy as np
from collections import defaultdict
import time
import heapq
import math
import torch
from collections import deque
import copy
from multiprocessing import Pool, cpu_count
from kingdomtreeworking import GameTree
from kingdomtreeworking import evaluate_best_leaf
from kingdomtreeworking import get_action_recommendation

def bfs_find_one_path(grid, base, resource_coords):
    rows, cols = len(grid), len(grid[0])
    resource_set = set(resource_coords)
    visited = {base}
    queue   = deque([base])
    parent  = {base: None}

    while queue:
        cur = queue.popleft()

        # check for resource
        if cur in resource_set:
            # reconstruct path
            path = []
            node = cur
            while node is not None:
                path.append(node)
                node = parent[node]
            return list(reversed(path))

        # explore neighbors
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = cur[0] + dr, cur[1] + dc
            nxt = (nr, nc)

            # bounds + walkable check
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr][nc] != 1:
                continue
            if cur == base and nxt in resource_set:
              continue

            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = cur
                queue.append(nxt)
    return []

def find_all_paths_to_resources(grid, base, resource_coords): #ERROR WHEN RESOURCE IS ADJACENT TO BASE
    working_grid = copy.deepcopy(grid)  # to not modify original grid
    paths = []

    while True: 
        path = bfs_find_one_path(working_grid, base, resource_coords)
        if not path:
            break  # No more paths available

        # Block the path (except base and resource)
        for cell in path[1:-1]:  # ignore base (first cell) and resource (last cell)
            r, c = cell
            working_grid[r][c] = 0  # mark as unwalkable

        path_rev = list(reversed(path))
        path = path[1:]
        path.extend(path_rev[1:])
        paths.append(path)

    return paths

#these variables will have to be changed with the actual code files
def caclulate_path_rates(all_paths, worker_speed = 1, harvest_time = 2, deposit_time = 1):
  if all_paths == []:
    return [] #idk we can do something here.[math.inf]
  gold_rates_per_path = []

  for path in all_paths:
    if path == []:
      gold_rates_per_path.append(math.inf)
      continue
    time_for_path = ((len(path)-4)/worker_speed) + harvest_time + deposit_time #-4 (-1 on a tile already, -1 don't have to "leave" resource, -2 for harvest and deposit steps)
    if time_for_path == 0:
      gold_rates_per_path.append(math.inf)
    else:
      gold_rates_per_path.append(time_for_path)


  return gold_rates_per_path


def pathfinding(worker_map, base_map, resource_map):
  #getting base coordinates
  coords = [(r, c) for r, row in enumerate(base_map) for c, val in enumerate(row) if val == 1] #change base_coord, #tesnor, tells you where the tensors are non 0
  if coords:
    base_coord = coords[0]  
  else:
    return [math.inf], None

  parsed_combined_map = [] #just add barracks map
  for r in range(len(worker_map)):
    row = []
    for c in range(len(worker_map[r])):
      row.append(1)
    parsed_combined_map.append(row)

  resource_coords = []
  for r in range(len(resource_map)):
      for c in range(len(resource_map[r])):
          if resource_map[r][c] == 1:
              resource_coords.append((r, c))

  if resource_coords == []:
    return [math.inf], [math.inf], base_coord

  bfs_paths = find_all_paths_to_resources(parsed_combined_map, base_coord, resource_coords)

  return caclulate_path_rates(bfs_paths), base_coord

def randomNum():
    return random.randint(0, 3), random.randint(0, 3)

def pad_to_corner(og_map, owner_id=0, size=16, pad_value=0):
    arr = np.array(og_map)
    h, w = arr.shape
    

    out = np.full((size, size), pad_value, dtype=arr.dtype)

    if owner_id == 0:
        out[:h, :w] = arr
    else:
        out[-h:, -w:] = arr

    return out

def find_t_rush(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    return (abs(x1 - x2) + abs(y1 - y2))/1 #T_rush is slightly 



def executeTwoTrees(user_state, enemy_state, optimal_paths):
    if enemy_state[1] < len(optimal_paths):
        e_current_path_rates = optimal_paths[:enemy_state[1].item()]
    else:
        e_current_path_rates = optimal_paths[::]
        for i in range(len(optimal_paths) - enemy_state[1]):
            e_current_path_rates.append(math.inf)

    t_rush = 22.0
    enemy_tree = GameTree(enemy_state, {0: e_current_path_rates}, optimal_paths, 1,  t_rush, runtime_limit=0.05)
    enemy_tree.build()
    max_mil_array_enemy = enemy_tree.get_mil_max()
    
    if user_state[1] < len(optimal_paths):
        u_current_path_rates = optimal_paths[:user_state[1].item()]
    else:
        u_current_path_rates = optimal_paths[::]
        for i in range(len(optimal_paths) - user_state[1]):
            u_current_path_rates.append(math.inf)

    u_current_path_rates = optimal_paths[:user_state[1].item()]
    our_tree = GameTree(user_state, {0: e_current_path_rates}, optimal_paths, 1,  t_rush, enemy_max_mil_arr=max_mil_array_enemy, runtime_limit=0.05)
    our_tree.build()

    best_leaf_recommendations = evaluate_best_leaf(our_tree, max_mil_array_enemy, t_rush, optimal_paths)
    output_tensor = get_action_recommendation(best_leaf_recommendations) #list of three tensors
    output_tensor = torch.cat(output_tensor, dim=0)

    return output_tensor


# for i in range(N): #should be N
#     u = user_db[i]
#     e = enemy_db[i]
#     out = executeTwoTrees(u, e, path_rates)

#     tree_outputs[i] = out
#print("Computed tree outputs:", tree_outputs.shape)


import torch
from multiprocessing import Pool, cpu_count
import os

# Globals for workers
_path_rates = None
_user_db = None
_enemy_db = None

# Called once per worker
def _init_worker(path_rates):
    global _path_rates, _user_db, _enemy_db
    _path_rates = path_rates
    _user_db = torch.load('user_db.pt', map_location='cpu')
    _enemy_db = torch.load('enemy_db.pt', map_location='cpu')

# Called with (i, j) index range
def process_chunk(start_end):
    start, end = start_end
    return [_worker_pair((_user_db[i], _enemy_db[i])) for i in range(start, end)]

# What each worker does
def _worker_pair(u_e):
    u, e = u_e
    return executeTwoTrees(u, e, _path_rates)

# Main script
if __name__ == '__main__':
    CHUNK_SIZE = 100 #1000
    print("Chunk size:", CHUNK_SIZE)
    print("Number of CPU cores:", cpu_count())

    # Map setup
    base_map = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0]
    ]
    resource_map = [
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]
    
    # Load data
    samples_db = torch.load('samples_db.pt')  # shape [N, 16]
    path_rates, base_coord = pathfinding(base_map, base_map, resource_map)
    print("Found paths and base...")
    
    # Slice and save once (so workers can load independently)
    user_db = samples_db[:, :8]
    enemy_db = samples_db[:, 8:]
    torch.save(user_db, 'user_db.pt')
    torch.save(enemy_db, 'enemy_db.pt')

    N = samples_db.size(0)
    #N = 10000 #only here to test

    chunks = [(i, min(i + CHUNK_SIZE, N)) for i in range(0, N, CHUNK_SIZE)]
    all_results = []


    print("Starting multiprocessing...")

    with Pool(
        processes=cpu_count(),
        initializer=_init_worker,
        initargs=(path_rates,)
    ) as pool:
        for i, chunk_result in enumerate(pool.imap_unordered(process_chunk, chunks), 1):
            all_results.extend(chunk_result)
            print(f"Finished chunk: {i}/{len(chunks)}")


    tree_outputs = torch.stack(all_results, dim=0)
    torch.save(tree_outputs, "tree_outputs.pt")
    print("Saved tree_outputs.pt")

