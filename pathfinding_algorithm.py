# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 12:22:49 2022

@author: Ana
"""
import numpy as np

def euclidian(snake, apple):
    return np.sqrt(np.sum(np.abs(np.subtract(snake, apple))))


def basic_moves(apple, snake_head_loc, full_snake, previous):
    """ 
        Finds pretty basic path to the apple
        Uses Euclidian distance and a few restrictions
    """
    snake = snake_head_loc
    
    
    distances = {'up':np.inf,
                 'left':np.inf,
                 'down':np.inf,
                 'right':np.inf}
    if snake[0] < 490 and previous != 'down' and [snake[0], snake[1]-10] not in full_snake:
        distances['up'] = euclidian(np.array([snake[0], snake[1]-10]), apple)
    if snake[1] > 10 and previous != 'right'and [snake[0]-10, snake[1]] not in full_snake:
        distances['left'] = euclidian(np.array([snake[0]-10, snake[1]]), apple)
    if snake[0] > 10 and previous != 'up'and [snake[0], snake[1]+10] not in full_snake:
        distances['down'] = euclidian(np.array([snake[0], snake[1]+10]), apple)
    if snake[1] < 790 and previous != 'left' and [snake[0]+10, snake[1]] not in full_snake:
        distances['right'] = euclidian(np.array([snake[0]+10, snake[1]]), apple)
    
    print(distances)
    
    move = [k for k, v in distances.items() if v == min(distances)]
    if len(move) > 1:
        if snake[1] == apple[1]-10 or snake[1] == apple[1]+10:
            if 'left' in move:
                return 'left'
            return 'right'
        elif snake[0] == apple[0]-10 or snake[0] == apple[0]+10:
            if 'up' in move:
                return 'up'
            return 'down'
    return min(distances, key=distances.get)



def astar():
    pass