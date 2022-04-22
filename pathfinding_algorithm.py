# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 12:22:49 2022

@author: Ana
"""
import numpy as np

def manhattan(snake, apple):
    return np.sqrt(np.sum(np.abs(np.subtract(snake, apple))))


def basic_moves(apple, snake_head_loc, full_snake, previous, size, x, y):
    """ 
        Finds pretty basic path to the apple
        Uses Manhattan distance and a few restrictions
    """
    snake = snake_head_loc
    
    
    distances = {'up':np.inf,
                 'left':np.inf,
                 'down':np.inf,
                 'right':np.inf}
    if snake[1] > 0 and previous != 'down' and [snake[0], snake[1]-size] not in full_snake:
        distances['up'] = manhattan(np.array([snake[0], snake[1]-size]), apple)
        
    if snake[0] > 0 and previous != 'right'and [snake[0]-size, snake[1]] not in full_snake:
        distances['left'] = manhattan(np.array([snake[0]-size, snake[1]]), apple)
        
    if snake[1] < y-size and previous != 'up'and [snake[0], snake[1]+size] not in full_snake:
        distances['down'] = manhattan(np.array([snake[0], snake[1]+size]), apple)
        
    if snake[0] < x-size and previous != 'left' and [snake[0]+size, snake[1]] not in full_snake:
        distances['right'] = manhattan(np.array([snake[0]+size, snake[1]]), apple)
    
    print(distances)
    
    move = [k for k, v in distances.items() if v == min(distances)]
    if len(move) > 1:
        if 'left' in move and snake[0] > size:
                return 'left'
        elif 'right' in move and snake[0] < x-size:
            return 'right'
        elif 'up' in move and snake[1] > size:
            return 'up'
        elif 'down' in move and snake[0] < y-size:
            return 'down'
    return min(distances, key=distances.get)



def astar():
    pass


def hamiltonian():
    pass