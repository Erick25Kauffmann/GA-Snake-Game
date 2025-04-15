import json
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool

from SnakeGame import *
from constants import *


# ===============================
# Función para cargar configuración desde un JSON
# ===============================
def load_config(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
    return config

# ===============================
# Función para escribir en un txt
# ===============================
def log_top_agents(generation, top_agents, board_size, log_filename="top_agents_log.txt"):
    with open(log_filename, "a") as f:
        f.write(f"--- Generación {generation} ---\n")
        for idx, agent in enumerate(top_agents):
            f.write(f"Agente {idx}:\n")
            f.write(f"  Fitness: {agent.fitness:.2f} | Avg Apples: {agent.avg_apples:.2f} | Avg Steps: {agent.avg_steps:.2f}\n")
            f.write("  Decision Table:\n")
            for (state, decision, steps, apples, quality) in agent.decision_table:
                # Convertir la matriz a cadena
                state_str = np.array2string(state, separator=",")
                f.write(f"    Steps: {steps:3d}, Apples: {apples:3d}, Decision: {decision}, Quality: {quality}\n")
                f.write(f"    {state_str}\n")
            f.write("\n")


# ===============================
# Función para calcular la distancia Manhattan
# ===============================
def manhattan_distance(p1, p2):
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])


def generate_contiguous_snake(board_size, snake_length, max_attempts=1000):
    """
    Genera una lista de celdas (tuplas (i,j)) que forman una serpiente continua
    de longitud snake_length en un tablero de tamaño board_size x board_size.
    Si no se encuentra en max_attempts, retorna None.
    """
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for attempt in range(max_attempts):
        # Elegir una celda de inicio al azar
        start = (random.randint(0, board_size-1), random.randint(0, board_size-1))
        snake = [start]
        valid = True
        while len(snake) < snake_length:
            head = snake[-1]
            neighbors = []
            for dx, dy in directions:
                nxt = (head[0] + dx, head[1] + dy)
                # Revisar que esté dentro del tablero y no haya sido usado aún
                if 0 <= nxt[0] < board_size and 0 <= nxt[1] < board_size and nxt not in snake:
                    neighbors.append(nxt)
            if not neighbors:
                valid = False
                break
            # Escoger aleatoriamente uno de los vecinos
            snake.append(random.choice(neighbors))
        if valid and len(snake) == snake_length:
            return snake
    return None

# ===============================
# Función para obtener los movimientos permitidos
# (Se evita la dirección opuesta a la actual, asumiendo que la serpiente no puede retroceder)
# ===============================
def valid_moves(game: SnakeGame):
    opposites = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
    moves = ["UP","DOWN","LEFT","RIGHT"]
    if game.direction in opposites and opposites[game.direction] in moves:
        moves.remove(opposites[game.direction])
    return moves

# ===============================
# Heurística para determinar movimiento 
# ===============================
def heuristic_move(game: SnakeGame, prob=0.2):
    moves = valid_moves(game)
    head = game.snake[0]
    apple = game.apple_position
    distances = {}
    for move in moves:
        if move == "UP":
            nxt = (head[0], head[1]-1)
        elif move == "DOWN":
            nxt = (head[0], head[1]+1)
        elif move == "LEFT":
            nxt = (head[0]-1, head[1])
        elif move == "RIGHT":
            nxt = (head[0]+1, head[1])
        if (nxt[0] < 0 or nxt[0] >= game.board_size or
            nxt[1] < 0 or nxt[1] >= game.board_size or nxt in game.snake):
            distances[move] = float('inf')
        else:
            distances[move] = manhattan_distance(nxt, apple)
    if random.random() < prob:
        return min(distances, key=distances.get)
    else:
        return random.choice(moves)
    

def is_safe_move(game: SnakeGame, move):
    head = game.snake[0]
    if move == "UP":
        nxt = (head[0], head[1]-1)
    elif move == "DOWN":
        nxt = (head[0], head[1]+1)
    elif move == "LEFT":
        nxt = (head[0]-1, head[1])
    elif move == "RIGHT":
        nxt = (head[0]+1, head[1])
    else:
        nxt = head
    # Verifica que no esté fuera del tablero y que no choque con la serpiente
    if nxt[0] < 0 or nxt[0] >= game.board_size or nxt[1] < 0 or nxt[1] >= game.board_size:
        return False
    if nxt in game.snake:
        return False
    return True

def improved_heuristic_move(game: SnakeGame, prob=1.0):
    moves = valid_moves(game)
    safe_moves = [m for m in moves if is_safe_move(game, m)]
    if not safe_moves:
        return None

    head = game.snake[0]
    apple = game.apple_position
    distances = {}
    visited = set(game.visited_positions[-10:]) if hasattr(game, "visited_positions") else set()

    for move in safe_moves:
        if move == "UP":
            nxt = (head[0], head[1]-1)
        elif move == "DOWN":
            nxt = (head[0], head[1]+1)
        elif move == "LEFT":
            nxt = (head[0]-1, head[1])
        elif move == "RIGHT":
            nxt = (head[0]+1, head[1])
        
        dist = manhattan_distance(nxt, apple)

        # Penalización si se aleja
        if dist > manhattan_distance(head, apple):
            dist += 5

        # Penalización si el movimiento lleva a una posición repetida (posible bucle)
        if nxt in visited:
            dist += 3

        # Bonus si va directamente a la manzana
        if nxt == apple:
            dist -= 10

        distances[move] = dist

    if random.random() < prob:
        return min(distances, key=distances.get)
    else:
        return random.choice(safe_moves)


def reconstruct_snake_from_board(board):
    head = None
    body = []
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i, j] == SNAKE_HEAD:
                head = (i, j)
            elif board[i, j] == SNAKE_BODY:
                body.append((i, j))
    if head is None:
        raise ValueError("No se encontró la cabeza de la serpiente en el tablero.")
    # Ordenamos el cuerpo según la cercanía a la cabeza (heurístico)
    body = sorted(body, key=lambda pos: manhattan_distance(pos, head))
    return [head] + body

def generate_valid_state(board_size, current_state, return_snake=False):
    snake_positions = [(i, j) for i in range(board_size) for j in range(board_size)
                       if current_state[i, j] in [SNAKE_HEAD, SNAKE_BODY]]
    snake_length = len(snake_positions)
    if snake_length < 2:
        snake_length = 2  # Longitud mínima razonable

    new_snake = generate_contiguous_snake(board_size, snake_length)

    # Si no se pudo generar o está vacía, usar posición por defecto
    if not new_snake:
        mid = board_size // 2
        if mid + 1 < board_size:
            new_snake = [(mid, mid), (mid, mid + 1)]
        else:
            new_snake = [(mid, mid), (mid, mid - 1)]

    state = np.zeros((board_size, board_size), dtype=int)
    state[new_snake[0]] = SNAKE_HEAD
    for pos in new_snake[1:]:
        state[pos] = SNAKE_BODY

    available = [(i, j) for i in range(board_size) for j in range(board_size) if (i, j) not in new_snake]
    if available:
        apple_pos = random.choice(available)
        state[apple_pos] = APPLE

    if return_snake:
        # Calcular dirección basada en los dos primeros segmentos
        direction = None
        if len(new_snake) >= 2:
            head, second = new_snake[0], new_snake[1]
            dx, dy = head[0] - second[0], head[1] - second[1]
            if dx == 1: direction = "DOWN"
            elif dx == -1: direction = "UP"
            elif dy == 1: direction = "RIGHT"
            elif dy == -1: direction = "LEFT"
        else:
            direction = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
        return state, new_snake, direction
    else:
        return state