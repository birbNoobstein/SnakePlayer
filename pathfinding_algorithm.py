# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 12:22:49 2022

@author: Ana
"""
import numpy as np
import heapq
import math
import pygame as pg
from collections import defaultdict, deque


def manhattan(snake, apple):
    return np.abs(snake - apple).sum()  # np.sum(np.abs(np.subtract(snake, apple)))
def euclidean(snake, apple):
    return np.sqrt(np.square(snake - apple).sum())
def diagonal(snake, apple):
    return np.abs(snake - apple).max()
def basic_moves(apple, snake_head_loc, full_snake, previous, size, x, y):
    """ 
        Finds pretty basic path to the apple
        Uses Manhattan distance and a few restrictions
    """
    snake = snake_head_loc

    distances = {'up': np.inf,
                 'left': np.inf,
                 'down': np.inf,
                 'right': np.inf}
    full = np.array(full_snake)
    snake_is_up = ((np.array([snake[0], snake[1] - size]) == full).sum(axis=1) == full.shape[1]).any()
    if snake[1] >= 0 and previous != 'down' and not snake_is_up:
        distances['up'] = manhattan(np.array([snake[0], snake[1] - size]), apple)
    snake_is_left = ((np.array([snake[0] - size, snake[1]]) == full).sum(axis=1) == full.shape[1]).any()
    if snake[0] >= 0 and previous != 'right' and not snake_is_left:
        distances['left'] = manhattan(np.array([snake[0] - size, snake[1]]), apple)
    snake_is_down = ((np.array([snake[0], snake[1] + size]) == full).sum(axis=1) == full.shape[1]).any()
    if snake[1] < y - size and previous != 'up' and not snake_is_down:
        distances['down'] = manhattan(np.array([snake[0], snake[1] + size]), apple)
    snake_is_right = ((np.array([snake[0] + size, snake[1]]) == full).sum(axis=1) == full.shape[1]).any()
    if snake[0] < x - size and previous != 'left' and not snake_is_right:
        distances['right'] = manhattan(np.array([snake[0] + size, snake[1]]), apple)

    # move = [k for k, v in distances.items() if v == min(distances)]
    # if len(move) > 1:
    #    if 'left' in move and snake[0] > size:
    #        return 'left'
    #    elif 'right' in move and snake[0] < x-size:
    #        return 'right'
    #    elif 'up' in move and snake[1] > size:
    #        return 'up'
    #    elif 'down' in move and snake[0] < y-size:
    #        return 'down'
    return min(distances, key=distances.get)


opposite = {"right": "left", "left": "right", "up": "down", "down": "up"}
def coords_add(size):
    return {"left": (-size, 0), "right": (size, 0), "up":(0, -size), "down": (0, size)}
def is_dead_end(full_snake, attempt_coords, x, y, size, apple): # TODO
    empty = (x /size) * (y/size) - len(full_snake)
    q = deque()
    q.append(tuple(attempt_coords))
    reachable = 0
    visited = set()
    visited.add(tuple(attempt_coords))
    while q:
        if len(visited) > empty * 0.25:
            return False
        cur = q.popleft()
        for move, i in [("left", (-size, 0)), ("right", (size, 0)), ("up", (0, -size)), ("down", (0, size))]:
            next = (cur[0] + i[0], cur[1] + i[1])
            if next not in visited and is_safe_move(full_snake, x, y, move, cur, size, apple)[0]:
                q.append(next)
                visited.add(next)
    return True

def is_snake(pos, full_snake):
    full = np.array(full_snake)
    if ((np.array(pos) == full).sum(axis=1) == full.shape[1]).any():  # any -> is snake
        return True
    return False
def is_outside_edge(pos, x, y):
    if not (0 <= pos[0] < x) or not (0 <= pos[1] < y):
        return True
    return False
def is_safe_move(full_snake, x, y, move, currloc, size, apple):
    # if move "move" is safe, from currloc location
    movloc = []
    for h, i in [("left", (-size, 0)), ("right", (size, 0)), ("up", (0, -size)), ("down", (0, size))]:
        if move == h:
            attempt = currloc + np.array(i)
            movloc = attempt
            nextx, nexty = tuple(attempt)
            if not (0 <= nextx < x) or not (0 <= nexty < y):
                return False, -1
            pos = np.array((nextx, nexty))
            full = np.array(full_snake)
            if ((np.array(pos) == full).sum(axis=1) == full.shape[1]).any(): #any -> is snake
                return False, -1
    return True, diagonal(movloc, apple)


def astar(apple, snake_head_loc, full_snake, previous, size, x_loc, y_loc):
    def reconstruct_path(cameFrom, cur, apple):
        total_path = [cur]
        while cur in cameFrom.keys():
            cur = cameFrom[cur]
            total_path.append(cur)
        diff = np.array(total_path[-2]) - np.array(total_path[-1])
        if diff[0] > 0 and previous != "left":
                return "right"
        if diff[0] < 0 and previous != "right":
                return "left"
        if diff[1] > 0 and previous != "up":
                return "down"
        if diff[1] < 0 and previous != "down":
                return "up"
        return previous
    ...
    for minmax in [0, 1]:
        if minmax == 0:
            app = tuple(apple)
            heuristic_fun = manhattan
            discovered_squares = []
            heapq.heappush(discovered_squares, (0, tuple(snake_head_loc)))
            came_from = {}
            cheapest_current_path = defaultdict(lambda: math.inf)  # default length for a missing key - unvisited node
            cheapest_current_path[tuple(snake_head_loc)] = 0
            heuristic_paths = defaultdict(lambda: math.inf)
            heuristic_paths[tuple(snake_head_loc)] = heuristic_fun(snake_head_loc, apple)
            full = np.array(full_snake)
            while discovered_squares:
                current = heapq.heappop(discovered_squares)[
                    1]  # node in discovered_squares having the lowest heuristic score value
                # current = current[1]
                if current == app:
                    # print("success")
                    return reconstruct_path(came_from, current, apple)
                coords = []  # (-size, 0),(size, 0),(0,-size),(0, size)
                if current[0] > 0:
                    coords.append((-size, 0))
                if current[0] < x_loc - size:
                    coords.append((size, 0))
                if current[1] > 0:
                    coords.append((0, -size))
                if current[1] < y_loc - size:
                    coords.append((0, size))
                for x, y in coords:
                    neighbour = (current[0] + x, current[1] + y)
                    neighbour_heuristic = cheapest_current_path[current] + size
                    if neighbour_heuristic < cheapest_current_path[neighbour]:
                        # This path to neighbor is better than any previous one. Record it!
                        came_from[neighbour] = current
                        cheapest_current_path[
                            neighbour] = neighbour_heuristic  # + heuristic_fun(apple, np.array(neighbour))
                        heuristic_paths[neighbour] = neighbour_heuristic + heuristic_fun(apple, np.array(neighbour))

                        h = [x[1] for x in discovered_squares if neighbour == x[1]]
                        if len(h) == 0:  # discovered_squares:
                            if not ((np.array(neighbour) == full).sum(axis=1) == full.shape[1]).any():
                                heapq.heappush(discovered_squares, (heuristic_paths[neighbour], neighbour))
        else:
            return avoidance_move(full_snake, x_loc, y_loc, previous, snake_head_loc, size, apple)
        #print("failure")
    # Open set is empty but goal was never reached
    return -1
def avoidance_move(full_snake, x_loc, y_loc, previous, snake_head_loc, size, apple):
    moves = []
    neigh = []
    for i in [-size, 0, size]:
        for j in [-size, 0, size]:
            neigh.append((snake_head_loc[0] + i, snake_head_loc[1] + j))

    for move in ["up", "down", "left", "right"]:
        moves.append((is_safe_move(full_snake, x_loc, y_loc, move, snake_head_loc, size, apple), move))

    moves = [x for x in moves if x[0][0]]
    #potentialMoves = []
    if previous == "up":
        if (is_snake(neigh[0], full_snake) and not is_snake(neigh[1], full_snake))\
                or (is_snake(neigh[6], full_snake) and not is_snake(neigh[7], full_snake)):
            moves = [m if not is_dead_end(full_snake, snake_head_loc + np.array(coords_add(size)[m[1]]), x_loc, y_loc, size, apple) else
                     ((False, None), "") for m in moves]
    elif previous == "down":
        if (is_snake(neigh[2], full_snake) and not is_snake(neigh[1], full_snake)) \
                or (is_snake(neigh[8], full_snake) and not is_snake(neigh[7], full_snake)):
            moves = [m if not is_dead_end(full_snake, snake_head_loc + np.array(coords_add(size)[m[1]]), x_loc, y_loc, size, apple) else
                     ((False, None), "") for m in moves]
    elif previous == "left":
        if (is_snake(neigh[0], full_snake) and not is_snake(neigh[3], full_snake)) \
                or (is_snake(neigh[2], full_snake) and not is_snake(neigh[5], full_snake)):
            moves = [m if not is_dead_end(full_snake, snake_head_loc + np.array(coords_add(size)[m[1]]), x_loc, y_loc, size, apple) else
                     ((False, None), "") for m in moves]
    elif previous == "right":
        if (is_snake(neigh[6], full_snake) and not is_snake(neigh[3], full_snake)) \
                or (is_snake(neigh[8], full_snake) and not is_snake(neigh[5], full_snake)):
            moves = [m if not is_dead_end(full_snake, snake_head_loc + np.array(coords_add(size)[m[1]]), x_loc, y_loc, size, apple) else
                     ((False, None), "") for m in moves]

    moves = [x for x in moves if x[0][0]]
    if len(moves) > 0:
        worstMove = max(moves, key=lambda x: x[0][1])
        print(worstMove[1])
        return worstMove[1]
    return -1
def hamiltonian():
    pass
