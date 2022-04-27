# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 12:22:49 2022

@author: Ana
"""
import random

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
    return {"left": (-size, 0), "right": (size, 0), "up": (0, -size), "down": (0, size)}


def dead_end_score(full_snake, attempt_coords, x, y, size, apple):  # TODO
    empty = (x / size) * (y / size) - len(full_snake)
    q = deque()
    q.append(tuple(attempt_coords))
    reachable = 0
    visited = set()
    visited.add(tuple(attempt_coords))
    must_be_reachable = len(full_snake)  # TODO
    while q:
        cur = q.popleft()
        for move, i in [("left", (-size, 0)), ("right", (size, 0)), ("up", (0, -size)), ("down", (0, size))]:
            next = (cur[0] + i[0], cur[1] + i[1])
            if next not in visited and is_safe_move(full_snake, x, y, move, cur, size, apple)[0]:
                q.append(next)
                visited.add(next)
    return len(visited)


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
    h = move
    i = coords_add(size)[move]
    attempt = currloc + np.array(i)
    movloc = attempt
    nextx, nexty = tuple(attempt)
    if not (0 <= nextx < x) or not (0 <= nexty < y):
        return False, -1
    pos = np.array((nextx, nexty))
    full = np.array(full_snake)
    if ((np.array(pos) == full).sum(axis=1) == full.shape[1]).any():  # any -> is snake
        return False, -1
    return True, euclidean(movloc, apple)


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
    for minmax in [0.0, 1.0]:
        if minmax == 0.0:
            app = tuple(apple)
            heuristic_fun = manhattan
            discovered_squares = []
            heapq.heappush(discovered_squares, (0.0, tuple(snake_head_loc)))
            came_from = {}
            cheapest_current_path = defaultdict(lambda: math.inf)  # default length for a missing key - unvisited node
            cheapest_current_path[tuple(snake_head_loc)] = 0.0
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
                    coords.append((-size, 0.0))
                if current[0] < x_loc - size:
                    coords.append((size, 0.0))
                if current[1] > 0:
                    coords.append((0.0, -size))
                if current[1] < y_loc - size:
                    coords.append((0.0, size))
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
        # print("failure")
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
    # potentialMoves = []
    if previous == "up":
        if (is_snake(neigh[0], full_snake) and not is_snake(neigh[1], full_snake)) \
                or (is_snake(neigh[6], full_snake) and not is_snake(neigh[7], full_snake)):
            moves = [
                ((m[0][0],
                  dead_end_score(full_snake, snake_head_loc + np.array(coords_add(size)[m[1]]), x_loc, y_loc, size,
                                 apple)), m[1]) for m in moves]
    elif previous == "down":
        if (is_snake(neigh[2], full_snake) and not is_snake(neigh[1], full_snake)) \
                or (is_snake(neigh[8], full_snake) and not is_snake(neigh[7], full_snake)):
            moves = [
                ((m[0][0],
                  dead_end_score(full_snake, snake_head_loc + np.array(coords_add(size)[m[1]]), x_loc, y_loc, size,
                                 apple)), m[1]) for m in moves]
    elif previous == "left":
        if (is_snake(neigh[0], full_snake) and not is_snake(neigh[3], full_snake)) \
                or (is_snake(neigh[2], full_snake) and not is_snake(neigh[5], full_snake)):
            moves = [
                ((m[0][0],
                  dead_end_score(full_snake, snake_head_loc + np.array(coords_add(size)[m[1]]), x_loc, y_loc, size,
                                 apple)), m[1]) for m in moves]
    elif previous == "right":
        if (is_snake(neigh[6], full_snake) and not is_snake(neigh[3], full_snake)) \
                or (is_snake(neigh[8], full_snake) and not is_snake(neigh[5], full_snake)):
            moves = [
                ((m[0][0],
                  dead_end_score(full_snake, snake_head_loc + np.array(coords_add(size)[m[1]]), x_loc, y_loc, size,
                                 apple)), m[1]) for m in moves]

    moves = [x for x in moves if x[0][0]]
    if len(moves) > 0:
        worstMove = max(moves, key=lambda x: x[0][1])
        print(worstMove)
        return worstMove[1]
    else:
        print("no good safe move found")
        for move in ["up", "down", "left", "right"]:
            moves.append((is_safe_move(full_snake, x_loc, y_loc, move, snake_head_loc, size, apple), move))
        return max(moves, key=lambda x: x[0][1])[1]
        ...

    # return -1


def gen_MST(size, x_loc, y_loc, random_factor=10, image=False):
    n = int((x_loc / size) / 2)
    m = int((y_loc / size) / 2)
    print(m, n)
    grid = np.full((m, n), 0.0, dtype="f, f")
    for i in range(m):
        for j in range(n):
            grid[i, j] = (i, j)
    start = (int(m / 2), int(n / 2))
    # print(grid, start)
    visited = set()
    visited.add(start)
    q = []
    heapq.heappush(q, (0, start))
    while q:
        d, loc = heapq.heappop(q)
        y, x = loc
        for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            cy, cx = y + d[0], x + d[1]
            if (cy, cx) not in visited:
                if 0 <= cy < m and 0 <= cx < n:
                    heapq.heappush(q,
                                   (random.random() * random_factor + manhattan(np.array(start), np.array((cy, cx))), (cy, cx)))
                    visited.add((cy, cx))
                    grid[cy, cx] = (y, x)
    cycle = np.full((m * 2, n * 2), -1.0, dtype="f, f")
    visited = set()
    visited.add(start)
    q = deque()
    q.append(start)
    while q:
        i, j = q.popleft()  # trenutno polje
        ds = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            cy, cx = i + d[0], j + d[1]  # sosedna polja
            if 0 <= cy < m and 0 <= cx < n and (cy, cx) not in visited:
                vy, vx = grid[cy, cx]
                if vy == i and vx == j:  # kaže na trenutno polje
                    ds.remove(d)
                    q.append((cy, cx))
                    visited.add((cy, cx))
                    diff = (np.array((cy, cx)) - np.array(tuple(grid[cy, cx])))  # določi od kje
                    if diff[0] == -1:  # iz spodaj
                        cycle[2 * cy + 2, 2 * cx + 1] = (2 * cy + 1, 2 * cx + 1)
                        cycle[2 * cy + 1, 2 * cx] = (2 * cy + 2, 2 * cx)
                    if diff[0] == 1:  # iz zgoraj
                        cycle[2 * cy - 1, 2 * cx] = (2 * cy, 2 * cx)
                        cycle[2 * cy, 2 * cx + 1] = (2 * cy - 1, 2 * cx + 1)
                    if diff[1] == -1:  # iz desne
                        cycle[2 * cy, 2 * cx + 2] = (2 * cy, 2 * cx + 1)
                        cycle[2 * cy + 1, 2 * cx + 1] = (2 * cy + 1, 2 * cx + 2)
                    if diff[1] == 1:  # iz leve
                        cycle[2 * cy + 1, 2 * cx - 1] = (2 * cy + 1, 2 * cx)
                        cycle[2 * cy, 2 * cx] = (2 * cy, 2 * cx - 1)
        for d in ds:  # kje neporabljeni premiki
            if d[0] == 1:  # spodaj
                if tuple(cycle[2 * i + 1, 2 * j]) == (-1, -1):
                    cycle[2 * i + 1, 2 * j] = (2 * i + 1, 2 * j + 1)
            if d[0] == -1:  # zgoraj
                if tuple(cycle[2 * i, 2 * j + 1]) == (-1, -1):
                    cycle[2 * i, 2 * j + 1] = (2 * i, 2 * j)
            if d[1] == 1:  # desne
                if tuple(cycle[2 * i + 1, 2 * j + 1]) == (-1, -1):
                    cycle[2 * i + 1, 2 * j + 1] = (2 * i, 2 * j + 1)
            if d[1] == -1:  # leve
                if tuple(cycle[2 * i, 2 * j]) == (-1, -1):
                    cycle[2 * i, 2 * j] = (2 * i + 1, 2 * j)
    if image:
        import matplotlib.pyplot as plt
        for i in range(cycle.shape[0]):
            for j in range(cycle.shape[1]):
                plt.plot([j, cycle[i, j][1]], [i, cycle[i, j][0]])
        plt.show()
    return cycle


hami_cycle = None

def diff_to_direction(diff):
    if diff[1] == -1:
        return "down"
    if diff[1] == 1:
        return "up"
    if diff[0] == -1:
        return "right"
    if diff[0] == 1:
        return "left"
def hamiltonian(apple, snake_head_loc, full_snake, previous, size, x_loc, y_loc):
    global hami_cycle
    if hami_cycle is None:
        hami_cycle = gen_MST(size, y_loc, x_loc)
    next_a = hami_cycle[int(snake_head_loc[0]/size), int(snake_head_loc[1]/size)]
    next_position = np.array(tuple(next_a)) * size
    diff = snake_head_loc - next_position

    dir = diff_to_direction(diff / size)
    return dir

if __name__ == "__main__":
    hamiltonian(None, np.array((0,0)), None, None, 10, 80, 50)

    hamiltonian(None, np.array((0,0)), None, None, 10, 80, 50)