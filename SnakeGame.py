import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool

from constants import *

# ===============================
# Clase SnakeGame
# ===============================
class SnakeGame:
    def __init__(self, board_size):
        self.board_size = board_size
        self.visited_positions = []
        self.reset()
    
    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        # Posición inicial de la serpiente: aleatoria
        head = (random.randint(0, self.board_size-1), random.randint(0, self.board_size-1))
        # Generar cuerpo: seleccionar una celda adyacente válida
        posibles = []
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            cell = (head[0]+dx, head[1]+dy)
            if 0 <= cell[0] < self.board_size and 0 <= cell[1] < self.board_size:
                posibles.append(cell)
        tail = random.choice(posibles) if posibles else head
        self.snake = [head, tail]  # tamaño 2
        # Dirección inicial calculada a partir de head y tail
        self.direction = self._calc_direction(head, tail)
        self.place_apple()
        self.update_board()
        self.steps = 0
        self.food = 0
        self.alive = True
        
    def _calc_direction(self, head, tail):
        dx = head[0] - tail[0]
        dy = head[1] - tail[1]
        if dx > 0:
            return "RIGHT"
        elif dx < 0:
            return "LEFT"
        elif dy > 0:
            return "DOWN"
        elif dy < 0:
            return "UP"
        return random.choice(["UP","DOWN","LEFT","RIGHT"])
    
    def place_apple(self):
        empty_cells = [(i,j) for i in range(self.board_size) for j in range(self.board_size) if (i,j) not in self.snake]
        if not empty_cells:
            return
        self.apple_position = random.choice(empty_cells)
    
    def update_board(self):
        self.board.fill(EMPTY)
        if hasattr(self, "apple_position"):
            self.board[self.apple_position] = APPLE
        if self.snake:
            head = self.snake[0]
            self.board[head] = SNAKE_HEAD
            for part in self.snake[1:]:
                self.board[part] = SNAKE_BODY
    
    def move(self, action):
        if not self.alive:
            return
        head_x, head_y = self.snake[0]
        if action == "UP":
            new_head = (head_x, head_y-1)
        elif action == "DOWN":
            new_head = (head_x, head_y+1)
        elif action == "LEFT":
            new_head = (head_x-1, head_y)
        elif action == "RIGHT":
            new_head = (head_x+1, head_y)
        else:
            new_head = (head_x, head_y)
        self.steps += 1
        if (new_head[0] < 0 or new_head[0] >= self.board_size or
            new_head[1] < 0 or new_head[1] >= self.board_size or
            new_head in self.snake):
            self.alive = False
            return
        self.snake.insert(0,new_head)
        if new_head == self.apple_position:
            self.food += 1
            self.place_apple()
        else:
            self.snake.pop()
        self.visited_positions.append(self.snake[0])
        if len(self.visited_positions) > 50:
            self.visited_positions.pop(0)

        self.update_board()