# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 12:22:49 2022

@author: Ana
"""
import numpy as np

def euclidian(snake, apple):
    return np.sqrt(np.sum(np.abs(np.subtract(snake, apple))))


def astar(apple, snake_head_loc, full_snake, previous):
    """ 
        Finds the best path to the apple
        Uses Euclidian distance
    """

    
    snake = snake_head_loc
    
    
    distances = {'up':np.inf,
                 'left':np.inf,
                 'down':np.inf,
                 'right':np.inf}
    if snake[0] < 490 and previous != 'down':
        distances['up'] = euclidian(np.array([snake[0], snake[1]-10]), apple)
    if snake[1] > 10 and previous != 'right':
        distances['left'] = euclidian(np.array([snake[0]-10, snake[1]]), apple)
    if snake[0] > 10 and previous != 'up':
        distances['down'] = euclidian(np.array([snake[0], snake[1]+10]), apple)
    if snake[1] < 790 and previous != 'left':
        distances['right'] = euclidian(np.array([snake[0]+10, snake[1]]), apple)
    
    #print(distances)        
    return min(distances, key=distances.get)
'''
    if move == 'up':
        snake = np.array([snake[0]-1, snake[1]])
    elif move == 'left':
        snake = np.array([snake[0], snake[1]-1])
    elif move == 'down':
        snake = np.array([snake[0]+1, snake[1]])
    elif move == 'right':
        snake = np.array([snake[0], snake[1]+1])
            
    grid[snake[0], snake[1]] = 2
    #print(snake, apple)
    yield move
    previous = move
    if snake[0] == apple[0] and snake[1] == apple[1]:
        break
    
'''
    




"""
grid = np.zeros((5, 7))
grid[4,6] = 2
#grid[3,4] = 2
grid[1,1] = 1
print(grid)
print(list(astar(grid, 1, np.array([4,6]), np.ones(2))))
"""