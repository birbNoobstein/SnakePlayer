# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 13:29:18 2022

@author: Ana
"""

# Load all the libraries needed for running the code chunks below
from selenium import webdriver
import time

from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pyautogui as pg
from PIL import Image

from astar_algorithm import astar

import sys; sys.path.append(".")


def new_apple_position(vertical_loc, horizontal_loc, playground):
    """ 
        Returns position of the apple
    """
    game_window = pg.screenshot(region=playground)
    red = (255, 0, 0)
    
    for v in range(vertical_loc.shape[0]):
        for h in range(horizontal_loc.shape[0]):
            if game_window.getpixel((int(horizontal_loc[h]),
                                     int(vertical_loc[v]))) == red:
                return [v, h]
            
            
def move(width, height, lvl, direction, past):
    """ 
        Clicks in front of the snake to change its direction
    """
    if lvl == 0 or direction != past:
        pg.click(width, height)
    else:
        time.sleep(0.01)
    

def draw_grid(interval):
    """ 
        Draws grid on game-window screenshot for easier interpretation
    """
    image = Image.open("playground.png")
    my_dpi=300
    
    fig=plt.figure(figsize=(float(image.size[0])/my_dpi,float(image.size[1])/my_dpi),dpi=my_dpi)
    ax=fig.add_subplot(111)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

    
    locx = plticker.MultipleLocator(base=interval)
    locy = plticker.MultipleLocator(base=interval)
    ax.xaxis.set_minor_locator(locx)
    ax.yaxis.set_minor_locator(locy)
    
    ax.grid(which='minor', axis='both', linestyle='-', color='white')
    ax.imshow(image)
    
            
    fig.savefig('plgrnd3.png', dpi=my_dpi)
    
    


def create_grid(game_window, vertical_loc, horizontal_loc):
    """ 
        Creates 2D-array that represents the game-window
        Checks colors in the middle of each square:
                                - if red (apple) --> gives value 1
                                - if green (snake) --> gives value 2
                                -otherwise value is 0
    """
    red = (255, 0, 0)
    green = (53,222,0)
    
    grid_values = np.zeros((vertical_loc.shape[0], horizontal_loc.shape[0]))
    
    
    for v in range(vertical_loc.shape[0]):
        for h in range(horizontal_loc.shape[0]):
            if game_window.getpixel((int(horizontal_loc[h]),
                                     int(vertical_loc[v]))) == red:
                grid_values[v, h] = 1
                apple = [v, h]
            elif game_window.getpixel((int(horizontal_loc[h]),
                                     int(vertical_loc[v]))) == green:
                grid_values[v, h] = 2
                snake = [v, h]
    
    
    return grid_values, snake, apple


def window_position(frame, lowpoints = [0, 0], initial=False):
    """ 
        Finds the position of the game-window (where window starts from left and up)
        Finds game-window width and height
    """
    background_color_dark = (0, 0, 0)
    background_color_light = (5, 5, 5)
    background_w, background_h = [], []
    
    for w in range(lowpoints[0], frame.width):
        for h in range(lowpoints[1], frame.height-40):
            if frame.getpixel((w, h)) >= background_color_dark and frame.getpixel((w, h)) <= background_color_light:
                background_w.append(w)
                background_h.append(h)
     
    position_left = min(background_w)
    position_up = min(background_h)
    width = max(background_w) - position_left + 1
    height = max(background_h) - position_up + 1
    
    return (position_left, position_up, width, height)
                

def play_game(path, snake_head, vertical_locations, horizontal_locations, playground):
    for e, p in enumerate(path):
        if p == 'up':
            move(horizontal_locations[snake_head[1]], 
                 playground[1]+vertical_locations[snake_head[0]-1], 
                 e, p, path[e-1])
            snake_head = [snake_head[0]-1, snake_head[1]]
        elif p == 'left':
            move(horizontal_locations[snake_head[1]-1], 
                 playground[1]+vertical_locations[snake_head[0]], 
                 e, p, path[e-1])
            snake_head = [snake_head[0], snake_head[1]-1]
        elif p == 'down':
            move(horizontal_locations[snake_head[1]], 
                 playground[1]+vertical_locations[snake_head[0]+1], 
                 e, p, path[e-1])
            snake_head = [snake_head[0]+1, snake_head[1]]                    
        else:
            move(horizontal_locations[snake_head[1]+1], 
                 playground[1]+vertical_locations[snake_head[0]], 
                 e, p, path[e-1])
            snake_head = [snake_head[0], snake_head[1]+1]
        print(p, snake_head)
        print(pg.position())
        time.sleep(0.04)
    

def start_game(url):
    """ 
        Opens game, finds the game-window position/dimention
    """
    game = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    game.maximize_window()
    game.get(url)

    time.sleep(2)
    pg.press('space')
    time.sleep(1)
    
    initial_frame = pg.screenshot()
    playground = window_position(initial_frame, lowpoints=[4, 40], initial=True)
    
    playground_scs = pg.screenshot(region=playground)
    playground_scs.save('playground.png')
    
    square_side = round(playground[3]/20)
    vertical_locations = np.array(range(round(square_side/2), playground[3], square_side)).astype(int)
    horizontal_locations = np.array(range(round(square_side/2), playground[2], square_side)).astype(int)
    
    #print(vertical_locations, horizontal_locations)
    #draw_grid(square_side)
    
    grid, snake, apple = create_grid(playground_scs, vertical_locations, horizontal_locations)
    
    snake_len = 1
    previous = None
    
    for i in range(6):
        path = list(astar(grid, snake_len, snake, apple, None))
        #pg.moveTo(playground[0]+playground[2]/2, playground[1]+playground[3]/2)
        play_game(path, snake, vertical_locations, horizontal_locations, playground)
        snake_len += 4
        snake = apple
        apple = new_apple_position(vertical_locations, horizontal_locations, playground)
        previous = path[-1]
    
if __name__ == '__main__':
    start_game("https://www.coolmathgames.com/0-snake/play")
    