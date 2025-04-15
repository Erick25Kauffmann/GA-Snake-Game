import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import threading
from multiprocessing.dummy import Pool as ThreadPool

from constants import *
from Functions import *
from SnakeGame import *

# ===============================
# Clase DecisionTableAgent
# ===============================
class DecisionTableAgent:
    def __init__(self, decision_table=None, generation=0):
        if decision_table is None:
            self.decision_table = []
        else:
            self.decision_table = decision_table
        self.generation = generation
        self.fitness = 0
        self.avg_apples = 0
        self.avg_steps = 0
        self.decision_table_lock = threading.Lock()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Eliminar el lock que no se puede copiar
        if 'decision_table_lock' in state:
            del state['decision_table_lock']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Al reestablecer el objeto, reinicializamos el lock
        self.decision_table_lock = threading.Lock()

    # Fase 1: Llenar la tabla con "games_for_agent" juegos, usando heurística (sin consultar la tabla)    
    def fill_initial_table(self, games_for_agent, board_size, use_threads=True, max_steps=10000):
        def run_game_wrapper(_):
            game = SnakeGame(board_size)
            while game.alive and game.steps < max_steps:
                prev_state = game.board.copy()
                move_choice = heuristic_move(game, prob=0.2)
                was_safe = is_safe_move(game, move_choice)
                game.move(move_choice)
                quality = GOOD if was_safe and game.alive else BAD
                return (prev_state, move_choice, game.steps, game.food, quality)

        if use_threads:
            pool = ThreadPool(games_for_agent)
            results = pool.map(run_game_wrapper, range(games_for_agent))
            pool.close()
            pool.join()
            with self.decision_table_lock:
                self.decision_table.extend(results)
        else:
            for _ in range(games_for_agent):
                self.run_game()


    # Fase 2: Durante la evaluación, para cada movimiento, se consulta la tabla; si no se halla, se usa heurística
    def decide_phase2(self, game: SnakeGame, prob=0.0, board_matching=0.2, force_learning=False):
        current_state = game.board.copy()
        total_cells = current_state.size
        threshold_value = board_matching * total_cells
        head = game.snake[0]
        apple = game.apple_position

        # Buscar acciones aprendidas válidas
        best_action = None
        best_match_score = float("inf")  # menor distancia es mejor

        for state, action, _, _, quality in self.decision_table:
            if quality != GOOD:
                continue
            diff = np.sum(current_state != state)
            if diff <= threshold_value:
                dx, dy = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}[action]
                predicted = (head[0] + dx, head[1] + dy)
                if is_safe_move(game, action):
                    score = manhattan_distance(predicted, apple)
                    if score < best_match_score:
                        best_match_score = score
                        best_action = action

        if best_action:
            return best_action
        else:
            if force_learning:
                return random.choice(valid_moves(game))
            else:
                # No hay aprendizaje útil, usar heurística simple
                return heuristic_move(game, prob=prob)



    def mutate(self, mutation_percent_board, mutation_percent_decision, board_size, mutation_new_cromosomes):
        num_chromosomes = len(self.decision_table)
        num_board = int(num_chromosomes * mutation_percent_board)
        num_decision = int(num_chromosomes * mutation_percent_decision)

        indices_board = random.sample(range(num_chromosomes), num_board) if num_chromosomes > 0 else []
        indices_decision = random.sample(range(num_chromosomes), num_decision) if num_chromosomes > 0 else []

        new_decision_table = []

        for i in range(num_chromosomes):
            state, decision, steps_count, apples_count, _ = self.decision_table[i]  # También traemos el quality

            if i in indices_board:
                dummy_state = np.zeros((board_size, board_size), dtype=int)
                state, snake, direction = generate_valid_state(board_size, dummy_state, return_snake=True)
                dummy_game = SnakeGame(board_size)
                dummy_game.board = state.copy()
                dummy_game.snake = snake
                dummy_game.direction = direction
                decision = random.choice(valid_moves(dummy_game))
            elif i in indices_decision:
                dummy_game = SnakeGame(board_size)
                dummy_game.board = state.copy()
                dummy_game.snake = reconstruct_snake_from_board(state)
                dummy_game.direction = dummy_game._calc_direction(dummy_game.snake[0], dummy_game.snake[1])
                decision = random.choice(valid_moves(dummy_game))

            # Evaluar calidad de la nueva decisión
            dummy_game = SnakeGame(board_size)
            dummy_game.board = state.copy()
            dummy_game.snake = reconstruct_snake_from_board(state)
            dummy_game.direction = dummy_game._calc_direction(dummy_game.snake[0], dummy_game.snake[1])
            head = dummy_game.snake[0]
            apple = dummy_game.apple_position
            next_pos = {
                "UP": (head[0], head[1] - 1),
                "DOWN": (head[0], head[1] + 1),
                "LEFT": (head[0] - 1, head[1]),
                "RIGHT": (head[0] + 1, head[1])
            }[decision]
            prev_dist = manhattan_distance(head, apple)
            post_dist = manhattan_distance(next_pos, apple)
            quality = GOOD if is_safe_move(dummy_game, decision) and post_dist < prev_dist else BAD

            new_decision_table.append((state, decision, steps_count, apples_count, quality))

        # Generar nuevos cromosomas (exploración)
        for _ in range(mutation_new_cromosomes):
            dummy_state = np.zeros((board_size, board_size), dtype=int)
            state, snake, direction = generate_valid_state(board_size, dummy_state, return_snake=True)
            dummy_game = SnakeGame(board_size)
            dummy_game.board = state.copy()
            dummy_game.snake = snake
            dummy_game.direction = direction
            decision = random.choice(valid_moves(dummy_game))

            head = dummy_game.snake[0]
            apple = dummy_game.apple_position
            next_pos = {
                "UP": (head[0], head[1] - 1),
                "DOWN": (head[0], head[1] + 1),
                "LEFT": (head[0] - 1, head[1]),
                "RIGHT": (head[0] + 1, head[1])
            }[decision]
            prev_dist = manhattan_distance(head, apple)
            post_dist = manhattan_distance(next_pos, apple)
            quality = GOOD if is_safe_move(dummy_game, decision) and post_dist < prev_dist else BAD

            new_decision_table.append((state, decision, 0, 0, quality))

        self.decision_table = new_decision_table



    @staticmethod
    def three_point_crossover(parent1, parent2):
        table1 = parent1.decision_table
        table2 = parent2.decision_table
        min_len = min(len(table1), len(table2))
        if min_len < 3:
            cp = random.randint(1, min_len) if min_len > 0 else 0
            new_table = table1[:cp] + table2[cp:]
            return DecisionTableAgent(new_table)
        cuts = sorted(random.sample(range(1, min_len), 3))
        new_table = []
        new_table.extend(table1[:cuts[0]])
        new_table.extend(table2[cuts[0]:cuts[1]])
        new_table.extend(table1[cuts[1]:cuts[2]])
        new_table.extend(table2[cuts[2]:min_len])
        return DecisionTableAgent(new_table)