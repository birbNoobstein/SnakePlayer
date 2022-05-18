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


def basic_moves(apple, snake_head_loc, full_snake, previous, size, x, y, game_board):
    """ 
        Finds pretty basic path to the apple
        Uses Manhattan distance and a few restrictions
    """
    diag = False
    mapping = coords_add(size)
    snake = snake_head_loc.copy()
    valid_moves = ["up", "down", "left", "right"]
    coords = []
    for move in ["up", "down", "left", "right"]:
        next_pos = np.array(mapping[move]) + snake
        if 0 <= next_pos[0] < x and 0 <= next_pos[1] < y and game_board[next_pos[0]//size, next_pos[1]//size] == 0:
            coords.append(next_pos)
        else:
            valid_moves.remove(move)
    if coords:
        cs = [manhattan(x, apple) for x in coords]
        min_ids = np.flatnonzero(cs == np.min(cs))
        if diag:
            if valid_moves[min_ids[0]] == previous and len(min_ids) > 1:
                return valid_moves[min_ids[1]]
        return valid_moves[min_ids[0]]
    else:
        return "dead"
opposite = {"right": "left", "left": "right", "up": "down", "down": "up"}


def coords_add(size):
    return {"left": (-size, 0), "right": (size, 0), "up": (0, -size), "down": (0, size),
            (-size, 0): "left", (size, 0): "right", (0, -size): "up", (0, size): "down"}


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


def astar(apple, snake_head_loc, full_snake, previous, size, x_loc, y_loc, game_board):
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
    for minmax in [0.0]:
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


def gen_MST(size, x_loc, y_loc, random_factor=10.0, image=False):
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
        plt.rcParams["figure.figsize"] = (4, 4)
        for i in range(cycle.shape[0]):
            for j in range(cycle.shape[1]):
                plt.plot([i, cycle[i, j][0]], [-j, -cycle[i, j][1]])
        #plt.savefig('hamiltonian_cycle.png')
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

def longest_skip(apple, snake_head, previous, size, game_board):
    moves_dict = coords_add(1)
    global hami_cycle
    dists = []
    moves = ["up", "down", "left", "right"]
    apple = np.array(apple)//size
    start = (np.array(snake_head)//size)
    valid_moves = []
    names = []
    for move in moves:
        if previous is None or move != opposite[previous]:
            offset = moves_dict[move]
            next = (start[0] + offset[0], start[1] + offset[1])
            if 0 <= next[0] < game_board.shape[0] and 0 <= next[1] < game_board.shape[1]:
                if game_board[next[0], next[1]] == 0:
                    valid_moves.append(next)
                    names.append(move)
    curr = tuple(hami_cycle[int(start[0]), int(start[1])])
    maxskip = diff_to_direction(start - np.array(curr))
    while game_board[int(curr[0]), int(curr[1])] == 0:
        if curr[0] == apple[0] and curr[1] == apple[1]:
            break
        try:
            n = valid_moves.index(curr)
            maxskip = names[n]
        except ValueError:
            pass
        curr = tuple(hami_cycle[int(curr[0]), int(curr[1])])
        #TRENUTNO: vsakič vzame največji skip
        #TODO: Dodaj skip če ta skip pomeni da je pot do jabolka krajša
    return maxskip


def find_valid_moves(cycle_1d, moves, moves_dict, previous, start, game_board):
    valid_cycle_indices = []
    for move in moves:
        if previous is None or move != opposite[previous]:
            offset = moves_dict[move]
            next = (start[0] + offset[0], start[1] + offset[1])
            if 0 <= next[0] < game_board.shape[0] and 0 <= next[1] < game_board.shape[1]:
                if previous is None or move != opposite[previous]:
                    index = np.where(np.array(list(map(lambda x: x == np.array(next).astype('f, f')[0], cycle_1d))))[0]
                    valid_cycle_indices.append(index)
    return valid_cycle_indices

def cycle_to_1d(cycle, start):
    cycle_1d = []
    element = np.array(start).astype('f, f')[0]
    while element not in cycle_1d:
        cycle_1d.append(element)
        element = cycle[int(element[0]), int(element[1])]
    return np.array(cycle_1d, dtype='object')

def skip_part(apple, snake_head, snake_full, previous, size, game_board):
    moves = ["up", "down", "left", "right"]
    moves_dict = coords_add(1)
    print('md:', moves_dict)
    cycle_1d = cycle_to_1d(hami_cycle, snake_head)
    valid_moves = find_valid_moves(cycle_1d, moves, moves_dict, previous, snake_head, game_board)
    names = []
    head = np.where(np.array(list(map(lambda x: x == np.array(tuple(snake_head)).astype('f, f')[0], cycle_1d))))[0][0]
    tail = np.where(np.array(list(map(lambda x: x == snake_full[-1].astype('f, f'), cycle_1d))))[0][0]
    print('H', head, cycle_1d[head])
    print('T', tail, cycle_1d[tail])
    curr = tuple(hami_cycle[snake_head[0], snake_head[1]])
    print('curr', curr)
    print('diff', snake_head - np.array([int(curr[0]), int(curr[1])]))
    maxskip = diff_to_direction(snake_head - np.array([int(curr[0]), int(curr[1])]))
    print('Hami\n', hami_cycle)
    print('1D\n', cycle_1d)
    print(game_board[int(curr[0]), int(curr[1])])

    while game_board[int(curr[0]), int(curr[1])] == 0:
        if curr[0] == apple[0] and curr[1] == apple[1]:
            break
        try:

            #print('n', position_to_validate, 'head', head, 'tail', tail)
            if curr > head and curr < tail or curr < head and curr > tail:
                maxskip = names[valid_moves.index[curr]]
        except ValueError:
            pass
        head = np.where(np.array(list(map(lambda x: x == np.array(tuple(curr)).astype('f, f')[0], cycle_1d))))[0][0]
        print(head)
        if len(snake_full) >= 2:
            tail = np.where(np.array(list(map(lambda x: x == snake_full[-2].astype('f, f'), cycle_1d))))[0][0]
        curr = tuple(hami_cycle[snake_head - np.array([int(curr[0]), int(curr[1])])])
    return maxskip

def hamiltonian(apple, snake_head_loc, full_snake, previous, size, x_loc, y_loc, game_board):
    global hami_cycle
    if hami_cycle is None:
        hami_cycle = gen_MST(size, y_loc, x_loc, random_factor=0.1, image=True)
    next_a = hami_cycle[int(snake_head_loc[0]/size), int(snake_head_loc[1]/size)]
    next_position = np.array(tuple(next_a)) * size
    diff = snake_head_loc - next_position
    direc = diff_to_direction(diff / size)
    #direc = skip_part(apple, snake_head_loc, full_snake, previous, size, game_board)
    direc = longest_skip(apple, snake_head_loc, previous, size, game_board)
    return direc



if __name__ == "__main__":
    hamiltonian(np.array((4, 8)), np.array((1,1)), np.array(np.array((1,1))), None, 10, 200, 200, np.zeros((10, 10)))
    #basic_moves(np.array((4, 8)), np.array((1,1)), np.array(np.array((1,1))), None, 10, 800, 500, np.zeros((10, 10)))