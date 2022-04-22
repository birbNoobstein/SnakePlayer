# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 18:08:53 2022

@author: Ana
"""

import pygame as pg
import sys
import random
import time
import numpy as np
import itertools
    
from pathfinding_algorithm import basic_moves

class SnackySnake:
    ''' 
        SNAKE GAME
        Takes argument pathfinder which defines the algorithm that will be
        choosing the path
    '''
    def __init__(self, pathfinder, fullscreen=False):
        #   Set display window
        self.pathfinder = pathfinder
        self.error_check = pg.init()
        pg.display.set_caption('Snacky Snake')
        self.x = int(np.floor(pg.display.get_desktop_sizes()[0][0]/20)*20) if fullscreen else 800
        self.y = int(np.floor(pg.display.get_desktop_sizes()[0][1]/20)*20) if fullscreen else 500
        self.game_window = pg.display.set_mode([self.x, self.y])
        
        #   Set positions
        self.snake_width = 20 if fullscreen else 10
        self.score = 0
        self.apple = [self.x-self.snake_width, self.y-self.snake_width] #[self.x/2-7*self.snake_width, self.y/2-5*self.snake_width]
        self.snake = [self.x/2, self.y/2]
        self.snake_full = [[self.x/2, self.y/2]]
        self.snake_tail = [self.x/2, self.y/2]
        
        self.snake_color = [pg.Color(53, 255, 0), pg.Color(28, 192, 192)]
        self.apple_color = pg.Color(255, 0, 0)
        self.path = None
        self.previous = None
        self.time_ctrl = pg.time.Clock()
        
    def set_apple(self):
        """ 
            Chooses coordinates to where apple moves from list of all positions
            that are not occupied by snake
        """
        pair_list = [l for l in list(itertools.product(range(0, self.x, self.snake_width),
                                                       range(0, self.y, self.snake_width))) if list(l) not in self.snake_full]
        self.apple = pair_list[random.randint(0, len(pair_list))]
        
    def set_path(self, previous):
        """ 
            Uses given algorithm to find the path to the apple
        """
        self.path = [self.pathfinder(self.apple, self.snake, self.snake_full, previous, self.snake_width, self.x, self.y)]
        
    def show_text(self, font, size, text, y):
        """ 
            Used to display endgame text, reason and score
        """
        font = pg.font.SysFont(font, size)
        surface = font.render(text, True, pg.Color(255, 255, 255))
        rect = surface.get_rect()
        rect.midtop = (self.x/2, y)
        self.game_window.blit(surface, rect)

        
    def end_game(self):
        """ 
            Checks if any game-ending constraints are satisfied,
            if yes --> terminates the game and returns the reason
        """
        for bodypart in self.snake_full[1:]:
            if self.snake[0] == bodypart[0] and self.snake[1] == bodypart[1]:
                return 'Crashed into yourself', True
        if self.snake[0] < 0 or self.snake[0] >= self.x or self.snake[1] < 0 or self.snake[1] >= self.y:
            return 'Crashed in to the wall', True
        return '/', False
    
    def game_over(self, reason):
        """ 
            Game termination, displays the YOU DIED text, death reason and 
            final length of the snake
        """
        self.show_text('constantia',
                       60, 
                       'YOU DIED', 
                       self.y/5)
        self.show_text('inkfree',
                       18, 
                       'Final length: ' + str(self.score), 
                       self.y/2)
        self.show_text('inkfree',
                       20, 
                       'Reason : ' + reason, 
                       self.y/1.6)
        pg.display.flip()
        time.sleep(3)
        pg.quit()
        sys.exit()
        
        
    def play(self):
        previous = None
        while True:
            pg.event.get() 
            self.set_path(previous)
            path = self.path
            
            for action in path:
                
                if action == 'up':
                    self.snake[1] -= self.snake_width
                elif action == 'down':
                    self.snake[1] += self.snake_width
                elif action == 'left':
                    self.snake[0] -= self.snake_width
                elif action == 'right':
                    self.snake[0] += self.snake_width
                    
                self.game_window.fill(pg.Color(0, 0, 0))
                self.snake_full.insert(0, self.snake.copy())
                if self.snake[0] == self.apple[0] and self.snake[1] == self.apple[1]:
                    self.score += 1
                    self.set_apple()
                    for e, bodypart in enumerate(self.snake_full):
                        pg.draw.rect(self.game_window, self.snake_color[e%2], pg.Rect(bodypart[0], bodypart[1], self.snake_width-2, self.snake_width-2))
                
                else :
                    self.snake_full = self.snake_full[0:-1]
                    for e, bodypart in enumerate(self.snake_full):
                        pg.draw.rect(self.game_window, self.snake_color[e%2], pg.Rect(bodypart[0], bodypart[1], self.snake_width-2, self.snake_width-2))
                
                print(previous, '-->', action)
                print(self.apple)
                print(self.snake)
                print(self.snake_full, '\n')
                    
                pg.draw.rect(self.game_window, self.apple_color, pg.Rect(self.apple[0], self.apple[1], self.snake_width, self.snake_width))
                
                previous = action
                pg.display.update()
                self.time_ctrl.tick(250)
                
                reason, end = self.end_game()
                if end:
                    self.game_window.fill(pg.Color(0, 0, 0))
                    self.game_over(reason)
                    
                
                    
if __name__ == '__main__':
    game = SnackySnake(basic_moves)
    game.play()