# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 12:22:49 2022

@author: Ana
"""
import numpy as np
import heapq
import math
import pygame as pg
from collections import defaultdict


def manhattan(snake, apple):
    return np.abs(snake - apple).sum()  # np.sum(np.abs(np.subtract(snake, apple)))


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


def next_safe_move(full_snake, x, y, prev, currloc, size):
    # TODO some safety checling when there is no clearcut path *sigh*
    for h, i in [("left", (-size, 0)), ("right", (size, 0)), ("up", (0, -size)), ("down", (0, size))]:
        attempt = currloc + np.array(i)
        nextx, nexty = tuple(attempt)
        if not (0 <= nextx < x) or not (0 <= nexty < y):
            continue
        pos = np.array((nextx, nexty))
        full = np.array(full_snake)
        if not ((np.array(pos) == full).sum(axis=1) == full.shape[1]).any():
            return h
    return "down"


def astar(apple, snake_head_loc, full_snake, previous, size, x_loc, y_loc):
    def reconstruct_path(cameFrom, cur, apple):
        total_path = [cur]
        while cur in cameFrom.keys():
            cur = cameFrom[cur]
            total_path.append(cur)
        diff = np.array(total_path[-2]) - np.array(total_path[-1])
        if diff[0] > 0:
            if previous != "left":
                return "right"
        if diff[0] < 0:
            if previous != "right":
                return "left"
        if diff[1] > 0:
            if previous != "up":
                return "down"
        if diff[1] < 0:
            if previous != "down":
                return "up"

    ...
    for minmax in [0, 1]:
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
                            if minmax == 0:
                                heapq.heappush(discovered_squares, (heuristic_paths[neighbour], neighbour))
                            else:
                                heapq.heappush(discovered_squares, (-heuristic_paths[neighbour], neighbour))

        print("failure")
    # Open set is empty but goal was never reached
    return -1


def hamiltonian():
    pass
