# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 18:08:53 2022

@author: Ana
"""

import pygame as pg
import sys
import random
import time
    
from astar_algorithm import astar

class SnackySnake:
    ''' 
        SNAKE GAME
    '''
    def __init__(self):
        self.error_check = pg.init()
        pg.display.set_caption('Snacky Snake')
        self.game_window = pg.display.set_mode([800, 500])
        self.score = 0
        self.apple = [200, 150]
        self.snake = [400, 250]
        self.snake_full = [[400, 250]]
        self.snake_tail = [400, 250]
        self.path = None
        self.snake_color = [pg.Color(53, 255, 0), pg.Color(28, 192, 192)]
        self.apple_color = pg.Color(255, 0, 0)
        self.previous = None
        self.time_ctrl = pg.time.Clock()
        
    def get_apple(self):
        return self.apple
    
    def set_apple(self):
        self.apple = [random.randrange(1, (800//10)) * 10, random.randrange(1, (500//10)) * 10]
        
    def set_path(self, previous):
        self.path = [astar(self.apple, self.snake, self.snake_full, previous)]
        
    def show_score(self, size, label, midgame=True):
        """ 
            Displays score in the bottom of the game-window during the game and
            close to the center at the end of the game
        """
        font = pg.font.SysFont('inkfree', size)
        score_surface = font.render(label + ' : ' + str(self.score), True, pg.Color(255, 255, 255))
        score_rect = score_surface.get_rect()
        if midgame == 1:
            score_rect.midtop = (745, 475)
        else:
            score_rect.midtop = (400, 350)
        self.game_window.blit(score_surface, score_rect)
        
    def death_reason(self, reason):
        """ 
            Displays the reason why you died
        """
        font = pg.font.SysFont('inkfree', 20)
        reason_surface = font.render('Reason : ' + reason, True, pg.Color(255, 255, 255))
        reason_rect = reason_surface.get_rect()
        reason_rect.midtop = (400, 300)
        self.game_window.blit(reason_surface, reason_rect)
        
    def end_game(self):
        """ 
            Checks if any game-ending constraints are satisfied,
            if yes --> terminates the game and returns the reason
        """
        for bodypart in self.snake_full[1:]:
            if self.snake[0] == bodypart[0] and self.snake[1] == bodypart[1]:
                return 'Crashed into yourself', True
        if self.snake[0] < 0 or self.snake[0] > 800 or self.snake[1] < 0 or self.snake[1] > 500:
            return 'Crashed in to the wall', True
        return '/', False
    
    def game_over(self, reason):
        """ 
            Game termination, displays the YOU DIED text, death reason and 
            final length of the snake
        """
        my_font = pg.font.SysFont('constantia', 60)
        game_over_surface = my_font.render('YOU DIED', True, pg.Color(255, 255, 255))
        game_over_rect = game_over_surface.get_rect()
        game_over_rect.midtop = (400, 125)
        self.game_window.fill(pg.Color(0, 0, 0))
        self.game_window.blit(game_over_surface, game_over_rect)
        self.show_score(18, 'Final length', midgame=False)
        self.death_reason(reason)
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
                    self.snake[1] -= 10
                elif action == 'down':
                    self.snake[1] += 10
                elif action == 'left':
                    self.snake[0] -= 10
                elif action == 'right':
                    self.snake[0] += 10
                    
                self.game_window.fill(pg.Color(0, 0, 0))
                self.snake_full.insert(0, self.snake.copy())
                if self.snake[0] == self.apple[0] and self.snake[1] == self.apple[1]:
                    #print('here', self.snake_full)
                    self.score += 1
                    self.set_apple()
                    for e, bodypart in enumerate(self.snake_full):
                        pg.draw.rect(self.game_window, self.snake_color[e%2], pg.Rect(bodypart[0], bodypart[1], 10, 10))
                
                else :
                    self.snake_full = self.snake_full[0:-1]
                    #print('snake', self.snake_full)
                    for e, bodypart in enumerate(self.snake_full):
                        pg.draw.rect(self.game_window, self.snake_color[e%2], pg.Rect(bodypart[0], bodypart[1], 10, 10))
                
                
                    
                pg.draw.rect(self.game_window, self.apple_color, pg.Rect(self.apple[0], self.apple[1], 10, 10))
                
                previous = action
                self.show_score(14, 'Length')
                pg.display.update()
                self.time_ctrl.tick(20)
                
                reason, end = self.end_game()
                if end:
                    #print(self.score)
                    self.game_window.fill(pg.Color(0, 0, 0))
                    self.game_over(reason)
                    
                
                    
if __name__ == '__main__':
    game = SnackySnake()
    game.play()