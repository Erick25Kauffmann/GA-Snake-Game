import json
import numpy as np
import random
import pygame
import matplotlib.pyplot as plt
import pandas as pd
import copy
import threading
from multiprocessing.dummy import Pool as ThreadPool

from SnakeGame import *
from constants import *

# ===============================
# Visualización con Pygame de los Top Agentes
# ===============================

def run_single_game(agent, board_size, max_steps=10000):
    """
    Ejecuta un único juego de Snake para el agente dado usando la función decide_phase2.
    Retorna una lista de frames (cada uno es una tupla: (board, steps, food)).
    """
    game = SnakeGame(board_size)
    frames = []
    while game.alive and game.steps < max_steps:
        action = agent.decide_phase2(game, prob=0, board_matching=0.2, force_learning=True)
        game.move(action)
        frames.append((game.board.copy(), game.steps, game.food, game.direction))
    return frames

def visualize_runs(runs, board_size, cell_size=20):
    """
    Visualiza en paralelo una única partida (run) para cada agente de la lista runs.
    Cada elemento de runs es una lista de frames del juego (board, steps, food).
    """
    pygame.init()
    num_agents = len(runs)
    max_frames = max(len(run) for run in runs)
    width = board_size * cell_size * num_agents
    height = board_size * cell_size + 50  # Espacio para mostrar contadores
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Visualización de juego único por Top Agente")
    font = pygame.font.SysFont(None, 24)
    clock = pygame.time.Clock()
    
    running = True
    frame_index = 0
    while running and frame_index < max_frames:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((0, 0, 0))
        for i in range(num_agents):
            run = runs[i]
            if frame_index < len(run):
                board, steps, food, direction = run[frame_index]
            else:
                board, steps, food, direction = run[-1]
            offset_x = i * board_size * cell_size
            for x in range(board_size):
                for y in range(board_size):
                    cell = board[y, x]
                    rect = pygame.Rect(offset_x + x * cell_size, y * cell_size, cell_size, cell_size)
                    if cell == EMPTY:
                        color = (40, 40, 40)
                    elif cell == SNAKE_HEAD:
                        color = (0, 255, 0)
                    elif cell == SNAKE_BODY:
                        color = (0, 200, 0)
                    elif cell == APPLE:
                        color = (255, 0, 0)
                    pygame.draw.rect(screen, color, rect)
            border_rect = pygame.Rect(offset_x, 0, board_size * cell_size, board_size * cell_size)
            pygame.draw.rect(screen, (255, 255, 255), border_rect, width=1)
            #info_text = font.render(f"Steps: {steps}  Apples: {food}", True, (255, 255, 255))
            #screen.blit(info_text, (offset_x, board_size * cell_size))
            text_steps = font.render(f"Steps: {steps}", True, (255,255,255))
            text_apples = font.render(f"Apples: {food}", True, (255,255,255))
            text_direction = font.render(f"Direction: {direction}", True, (255,255,255))
            screen.blit(text_steps, (i * board_size * cell_size, board_size * cell_size))
            screen.blit(text_apples, (i * board_size * cell_size, board_size * cell_size + 20))
            screen.blit(text_direction, (i * board_size * cell_size, board_size * cell_size + 60))
        pygame.display.flip()
        clock.tick(8)
        frame_index += 1
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        pygame.display.flip()
        clock.tick(10)
    pygame.quit()