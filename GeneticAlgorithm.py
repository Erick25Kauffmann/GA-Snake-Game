import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import copy
from multiprocessing.dummy import Pool as ThreadPool

from DecisionTableAgent import *
from SnakeGame import *
from Functions import *
from constants import *

# ===============================
# Clase GeneticAlgorithm
# ===============================
class GeneticAlgorithm:
    def __init__(self, config):
        self.board_size = config["board_size"]
        self.population_size = config["population_size"]
        self.generations = config["generations"]
        self.crossover_percentage = config["crossover_percentage"]
        self.mutation_percentage = config["mutation_percentage"]  # Usado para determinar el balance entre crossover y mutation
        self.elite_percentage = config["elite_percentage"]
        self.top_visualization = config["top_visualization"]
        # Nuevos parámetros
        self.games_for_agent = config.get("games_for_agent", 5)
        self.games_per_agent = config.get("games_per_agent", 20)
        self.bonus = config.get("bonus", 300)
        self.pen1 = config.get("pen1", 3)
        self.pen2 = config.get("pen2", 50)
        self.pen3 = config.get("pen3", 100)
        self.mt_value = config.get("mt_value", 50)
        self.mutation_percent_decision = config.get("mutation_percent_decision", 0.3)
        self.mutation_percent_board = config.get("mutation_percent_board", 0.1)
        self.mutation_new_cromosomes = config.get("mutation_new_cromosomes", 15)
        self.board_matching = config.get("board_matching", 0.2)
        self.pen_inactividad_alimenticia = config.get("pen_inactividad_alimenticia", 50)
        self.max_table_size = config.get("max_table_size", 2000)
        self.use_multithreading = config.get("use_multithreading", True)
        
        # Inicializar población
        self.population = [DecisionTableAgent() for _ in range(self.population_size)]
        self.fitness_history_avg = []
        self.fitness_history_max = []
        self.apple_count_history_avg = []
        self.apple_count_history_max = []
        self.fitness_distribution = []
        self.best_agents_by_gen = []
        
        # Fase 1: Llenar la tabla de cada agente con games_for_agent juegos
        for agent in self.population:
            agent.fill_initial_table(self.games_for_agent, self.board_size, use_threads=self.use_multithreading)
    
    
    # Fase 2: Evaluar un agente con games_per_agent juegos (paralelizando si use_multithreading está activado)
    def evaluate_agent(self, agent):
        def run_one_game(_):
            game = SnakeGame(self.board_size)
            steps = 0
            apples = 0
            inactividad = 0
            visited_decisions = []

            while game.alive:
                current_state = game.board.copy()
                move = agent.decide_phase2(game, prob=0.0, board_matching=self.board_matching, force_learning=True)
                prev_apples = game.food
                game.move(move)
                steps += 1

                # Evaluar calidad de decisión
                new_apples = game.food
                gained = new_apples > prev_apples
                quality = GOOD if is_safe_move(game, move) and gained else BAD
                visited_decisions.append((current_state, move, steps, new_apples, quality))

                if gained:
                    apples += 1
                    inactividad = 0
                else:
                    inactividad += 1

                if inactividad >= self.pen_inactividad_alimenticia:
                    break

            fitness = (apples * self.bonus) - (steps * self.pen1) - (inactividad * self.pen2)
            return fitness, apples, steps, visited_decisions

        # Crear pool de threads
        pool = ThreadPool(self.games_per_agent)
        results = pool.map(run_one_game, range(self.games_per_agent))
        pool.close()
        pool.join()

        # Consolidar resultados
        total_fitness = 0
        total_apples = 0
        total_steps = 0
        for fitness, apples, steps, decisions in results:
            total_fitness += fitness
            total_apples += apples
            total_steps += steps
            agent.decision_table.extend(decisions)

        # Limpiar tabla si se desborda (eliminar BADs)
        if len(agent.decision_table) > self.max_table_size:
            agent.decision_table = [entry for entry in agent.decision_table if entry[4] == GOOD]

        avg_fitness = total_fitness / self.games_per_agent
        avg_apples = total_apples / self.games_per_agent
        avg_steps = total_steps / self.games_per_agent

        agent.fitness = avg_fitness
        agent.avg_apples = avg_apples
        agent.avg_steps = avg_steps

        return avg_fitness, avg_apples, avg_steps


    def evaluate_population(self):
        if self.use_multithreading:
            # Evaluar todos los agentes en paralelo con ThreadPool
            pool = ThreadPool(self.population_size)
            results = pool.map(self.evaluate_agent, self.population)
            pool.close()
            pool.join()
            fitness_values = [f for f, _, _ in results]
            apples_values = [a for _, a, _ in results]
            return [fitness_values, apples_values]
        else:
            # Evaluación secuencial
            fitness_values = []
            apples_values = []
            for agent in self.population:
                f, a = self.evaluate_agent(agent)
                fitness_values.append(f)
                apples_values.append(a)
            return [fitness_values, apples_values]
    
    def selection(self, fitness_values):
        num_elite = max(1, int(self.elite_percentage * self.population_size))
        sorted_indices = np.argsort(fitness_values)[::-1]
        elite_agents = [self.population[i] for i in sorted_indices[:num_elite]]
        return elite_agents
    
    def create_next_generation(self, elite_agents, current_gen):
        new_population = elite_agents.copy()
        remainder = self.population_size - len(new_population)
        # Definimos que crossover_percentage se utiliza para determinar cuántos hijos se crean por cruce
        num_cross = int(remainder * self.crossover_percentage)
        num_mutation = int(remainder * self.mutation_percentage)
        #num_mutation = remainder - num_cross
        
        # Generar hijos mediante cruce
        for _ in range(num_cross):
            parent1, parent2 = random.sample(elite_agents, 2)
            child = DecisionTableAgent.three_point_crossover(parent1, parent2)
            child.generation = current_gen
            new_population.append(child)
        # Generar hijos mediante mutación de clones de individuos elite
        for _ in range(num_mutation):
            parent = random.choice(new_population)
            #child = copy.deepcopy(parent)
            #child.mutate(self.mutation_percent_decision, self.mutation_percent_board, self.board_size, self.mutation_new_cromosomes)
            #child.generation = current_gen
            #new_population.append(child)
            parent.mutate(self.mutation_percent_decision, self.mutation_percent_board, self.board_size, self.mutation_new_cromosomes)
        self.population = new_population

    
    def run(self):
        for generation in range(self.generations):
            pop_stats = self.evaluate_population()
            fitness_values = pop_stats[0]
            apple_values = pop_stats[1]
            avg_fit = np.mean(fitness_values)
            max_fit = np.max(fitness_values)
            avg_apples = np.mean(apple_values)
            max_apples = np.max(apple_values)
            self.fitness_history_avg.append(avg_fit)
            self.fitness_history_max.append(max_fit)
            self.apple_count_history_avg.append(avg_apples)
            self.apple_count_history_max.append(max_apples)
            self.fitness_distribution.append(fitness_values)
            
            print(f"Generación {generation}: fitness promedio = {avg_fit:.2f}, fitness máximo = {max_fit:.2f}, promedio manzanas = {avg_apples:.2f}, máximo manzanas = {max_apples:.2f}")
            
            best_index = np.argmax(fitness_values)
            best_agent = copy.deepcopy(self.population[best_index])
            best_agent.generation = generation
            self.best_agents_by_gen.append(best_agent)
            # Seleccionar y registrar los top agentes para esta generación
            top_agents = self.selection(fitness_values)
            #log_top_agents(generation, top_agents, self.board_size)
            elite_agents = self.selection(fitness_values)
            for agent in elite_agents:
                agent.generation = generation
            self.create_next_generation(elite_agents, generation)
        return (self.fitness_history_avg, self.fitness_history_max, 
                self.apple_count_history_avg, self.apple_count_history_max,
                self.fitness_distribution)