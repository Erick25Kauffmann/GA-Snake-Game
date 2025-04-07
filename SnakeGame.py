import json
import numpy as np
import random
import pygame
import matplotlib.pyplot as plt
import pandas as pd
import copy

# Constantes para la codificación del tablero
EMPTY = 0
SNAKE_HEAD = 1
SNAKE_BODY = 2
APPLE = 3

# Función para leer el archivo de configuración JSON
def load_config(config_file):
    with open(config_file, "r") as f:
        config = json.load(f)
    return config

# ------------------------------
# Clase que representa el juego Snake
# ------------------------------
class SnakeGame:
    def __init__(self, board_size):
        self.board_size = board_size
        self.reset()
        
    def reset(self):
        # Inicializa el tablero como una matriz de ceros
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        # La serpiente comienza en el centro del tablero
        start = (self.board_size // 2, self.board_size // 2)
        self.snake = [start]  # La lista representa el cuerpo de la serpiente; el primer elemento es la cabeza
        self.direction = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
        self.place_apple()
        self.update_board()
        self.steps = 0
        self.food = 0
        self.alive = True
        
    def place_apple(self):
        # Coloca la manzana en una celda vacía aleatoria
        empty_cells = list(zip(*np.where(self.board == EMPTY)))
        if not empty_cells:
            return
        self.apple_position = random.choice(empty_cells)
        
    def update_board(self):
        # Limpia el tablero y coloca la manzana y la serpiente
        self.board.fill(EMPTY)
        if hasattr(self, "apple_position"):
            self.board[self.apple_position] = APPLE
        if self.snake:
            head = self.snake[0]
            self.board[head] = SNAKE_HEAD
            for part in self.snake[1:]:
                self.board[part] = SNAKE_BODY
        
    def move(self, action):
        # Realiza el movimiento según la acción ("UP", "DOWN", "LEFT", "RIGHT")
        if not self.alive:
            return
        
        head_x, head_y = self.snake[0]
        if action == "UP":
            new_head = (head_x, head_y - 1)
        elif action == "DOWN":
            new_head = (head_x, head_y + 1)
        elif action == "LEFT":
            new_head = (head_x - 1, head_y)
        elif action == "RIGHT":
            new_head = (head_x + 1, head_y)
        else:
            new_head = (head_x, head_y)
        
        self.steps += 1
        
        # Verifica colisión con paredes
        if (new_head[0] < 0 or new_head[0] >= self.board_size or 
            new_head[1] < 0 or new_head[1] >= self.board_size):
            self.alive = False
            return
        
        # Verifica colisión consigo misma
        if new_head in self.snake:
            self.alive = False
            return
        
        # Actualiza la posición de la serpiente
        self.snake.insert(0, new_head)
        
        # Si la manzana es comida, aumenta el contador y coloca una nueva
        if new_head == self.apple_position:
            self.food += 1
            self.place_apple()
        else:
            # Elimina la cola si no se comió manzana
            self.snake.pop()
        
        self.update_board()

# ------------------------------
# Clase Agente basado en Tabla de Decisión
# ------------------------------
class DecisionTableAgent:
    def __init__(self, chromosome=None, generation=0):
        # Para simplificar, el cromosoma es un vector de 4 pesos (para UP, DOWN, LEFT, RIGHT)
        if chromosome is None:
            self.chromosome = np.random.rand(4)  # Pesos aleatorios
        else:
            self.chromosome = chromosome
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.generation = generation
        self.fitness = 0
        self.history = []
        
    def decide(self, game: SnakeGame):
        # El agente evalúa su entorno de forma muy simplificada:
        # Se observa la posición de la manzana respecto a la cabeza y se penalizan movimientos que colisionen.
        head = game.snake[0]
        apple = game.apple_position
        
        dx = apple[0] - head[0]
        dy = apple[1] - head[1]
        
        scores = np.copy(self.chromosome)
        if dx < 0:
            scores[self.actions.index("LEFT")] += abs(dx)
        elif dx > 0:
            scores[self.actions.index("RIGHT")] += abs(dx)
        if dy < 0:
            scores[self.actions.index("UP")] += abs(dy)
        elif dy > 0:
            scores[self.actions.index("DOWN")] += abs(dy)
        
        # Penaliza movimientos que llevarían a colisión
        for i, action in enumerate(self.actions):
            head_x, head_y = head
            if action == "UP":
                new_pos = (head_x, head_y - 1)
            elif action == "DOWN":
                new_pos = (head_x, head_y + 1)
            elif action == "LEFT":
                new_pos = (head_x - 1, head_y)
            elif action == "RIGHT":
                new_pos = (head_x + 1, head_y)
            else:
                new_pos = head
                
            if (new_pos[0] < 0 or new_pos[0] >= game.board_size or 
                new_pos[1] < 0 or new_pos[1] >= game.board_size or
                new_pos in game.snake):
                scores[i] -= 100  # Penalización fuerte
        return self.actions[np.argmax(scores)]
    
    def mutate(self, mutation_probability):
        # Mutación: para cada peso, con probabilidad mutation_probability se añade un valor pequeño aleatorio
        for i in range(len(self.chromosome)):
            if random.random() < mutation_probability:
                self.chromosome[i] += np.random.normal(0, 0.1)
    
    @staticmethod
    def crossover(parent1, parent2):
        # Cruce de un punto: se combina parte del cromosoma de cada padre
        child_chromosome = np.copy(parent1.chromosome)
        crossover_point = random.randint(1, len(parent1.chromosome)-1)
        child_chromosome[crossover_point:] = parent2.chromosome[crossover_point:]
        return DecisionTableAgent(child_chromosome)

# ------------------------------
# Clase para el Algoritmo Genético
# ------------------------------
class GeneticAlgorithm:
    def __init__(self, config):
        self.board_size = config["board_size"]
        self.population_size = config["population_size"]
        self.generations = config["generations"]
        self.crossover_probability = config["crossover_probability"]
        self.mutation_probability = config["mutation_probability"]
        self.elite_percentage = config["elite_percentage"]
        self.top_visualization = config["top_visualization"]
        
        self.population = [DecisionTableAgent() for _ in range(self.population_size)]
        self.fitness_history_avg = []
        self.fitness_history_max = []
        self.fitness_distribution = []  # Lista con los fitness de cada generación

        self.best_agents_by_gen = []
        
    def evaluate_agent(self, agent, max_steps=200):
        # Simula una partida y calcula el fitness:
        # fitness = (comida x 10) + pasos − (muerte prematura x 50)
        game = SnakeGame(self.board_size)
        agent.history = []
        current_gen = agent.generation
        while game.alive and game.steps < max_steps:
            action = agent.decide(game)
            game.move(action)
            partial_fitness = (game.food * 10) + game.steps
            agent.history.append((game.board.copy(), partial_fitness, current_gen))
        premature_death = 1 if game.food == 0 else 0
        final_fitness = (game.food * 10) + game.steps - (premature_death * 50)
        agent.fitness = final_fitness
        if len(agent.history) == 0:
            agent.history.append((game.board.copy(), final_fitness, current_gen))
        else:
            last_board, _, _ = agent.history[-1]
            agent.history[-1] = (last_board.copy(), final_fitness, current_gen)
        return final_fitness
    
    def evaluate_population(self):
        fitness_values = []
        for agent in self.population:
            fitness = self.evaluate_agent(agent)
            fitness_values.append(fitness)
        return fitness_values
    
    def selection(self, fitness_values):
        # Selección por élite: se selecciona el porcentaje configurado de los mejores individuos
        num_elite = max(1, int(self.elite_percentage * self.population_size))
        sorted_indices = np.argsort(fitness_values)[::-1]  # Orden descendente
        elite_agents = [self.population[i] for i in sorted_indices[:num_elite]]
        return elite_agents
    
    def create_next_generation(self, elite_agents, current_gen):
        new_population = elite_agents.copy()
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(elite_agents, 2)
            if random.random() < self.crossover_probability:
                child = DecisionTableAgent.crossover(parent1, parent2)
            else:
                child = copy.deepcopy(parent1)
            child.mutate(self.mutation_probability)
            child.generation = current_gen
            new_population.append(child)
        self.population = new_population
    
    def run(self):
        for generation in range(self.generations):
            fitness_values = self.evaluate_population()
            avg_fit = np.mean(fitness_values)
            max_fit = np.max(fitness_values)
            self.fitness_history_avg.append(avg_fit)
            self.fitness_history_max.append(max_fit)
            self.fitness_distribution.append(fitness_values)
            
            print(f"Generación {generation}: fitness promedio = {avg_fit:.2f}, fitness máximo = {max_fit:.2f}")

            # Seleccionamos el mejor agente de esta generación
            best_index = np.argmax(fitness_values)
            best_agent = copy.deepcopy(self.population[best_index])
            # Ajustamos su generación para que coincida con 'generation'
            best_agent.generation = generation
            # Guardamos al mejor agente (con su historial)
            self.best_agents_by_gen.append(best_agent)
            
            elite_agents = self.selection(fitness_values)
            for agent in elite_agents:
                agent.generation = generation
            self.create_next_generation(elite_agents, generation)
        return self.fitness_history_avg, self.fitness_history_max, self.fitness_distribution

# ------------------------------
# Visualización en Pygame de los Top 3 Agentes
# ------------------------------
def visualize_top_agents(agents, board_size, cell_size=20):
    pygame.init()
    # Ancho: tantos tableros como agentes
    width = board_size * cell_size * len(agents)
    height = board_size * cell_size + 50
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Visualización de Top Agentes")
    font = pygame.font.SysFont(None, 24)
    clock = pygame.time.Clock()
    
    # Calculamos cuántos "pasos" máximos hay entre todos los agentes
    max_length = max(len(agent.history) for agent in agents)
    
    running = True
    step_index = 0
    
    while running and step_index < max_length:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.fill((0, 0, 0))
        
        for i, agent in enumerate(agents):
            # Si este agente aún tiene historial en este step, lo usamos.
            # Si no, usamos el último estado (murió antes).
            if step_index < len(agent.history):
                board, current_fitness, _ = agent.history[step_index]
            else:
                board, current_fitness, _ = agent.history[-1]
            
            # Dibujamos el tablero en la columna i
            for x in range(board_size):
                for y in range(board_size):
                    cell_value = board[y, x]
                    rect = pygame.Rect(
                        i * board_size * cell_size + x * cell_size, 
                        y * cell_size, 
                        cell_size, 
                        cell_size
                    )
                    if cell_value == 0:   # EMPTY
                        color = (40, 40, 40)
                    elif cell_value == 1: # SNAKE_HEAD
                        color = (0, 255, 0)
                    elif cell_value == 2: # SNAKE_BODY
                        color = (0, 200, 0)
                    elif cell_value == 3: # APPLE
                        color = (255, 0, 0)
                    pygame.draw.rect(screen, color, rect)
            
            # Dibujamos un borde blanco alrededor
            border_rect = pygame.Rect(
                i * board_size * cell_size, 
                0, 
                board_size * cell_size, 
                board_size * cell_size
            )
            pygame.draw.rect(screen, (255, 255, 255), border_rect, width=1)
            
            # Mostramos el fitness y la generación
            text = font.render(
                f"Fit: {current_fitness}", 
                True, 
                (255, 255, 255)
            )
            screen.blit(text, (i * board_size * cell_size, board_size * cell_size))
        
        pygame.display.flip()
        clock.tick(8)  # Controla la velocidad de la animación
        step_index += 1
    
    # Mantener la ventana al terminar la animación
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        pygame.display.flip()
        clock.tick(10)
    
    pygame.quit()


# ------------------------------
# Visualización de Resultados con Matplotlib y Pandas
# ------------------------------
def plot_results(fitness_history_avg, fitness_history_max, fitness_distribution):
    generations = list(range(len(fitness_history_avg)))
    
    # Gráfico de líneas: evolución del fitness promedio y máximo
    plt.figure("Evolución del Fitness")
    plt.plot(generations, fitness_history_avg, label="Fitness Promedio")
    plt.plot(generations, fitness_history_max, label="Fitness Máximo")
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.title("Evolución del Fitness")
    plt.legend()
    plt.grid(True)
    #plt.show()
    
    # Gráfico de barras: distribución del fitness en la última generación
    plt.figure("Distribución del Fitness (Última Generación)")
    plt.bar(range(len(fitness_distribution[-1])), fitness_distribution[-1])
    plt.xlabel("Individuo")
    plt.ylabel("Fitness")
    plt.title("Distribución del Fitness en la Última Generación")
    #plt.show()
    
    # Boxplot para visualizar la distribución por generación
    df = pd.DataFrame(fitness_distribution).T
    plt.figure("Boxplot del Fitness por Generación")
    df.boxplot()
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.title("Boxplot del Fitness por Generación")
    #plt.show()
    
    # Gráfico adicional sugerido: histograma del fitness en la última generación
    plt.figure("Histograma del Fitness (Última Generación)")
    plt.hist(fitness_distribution[-1], bins=10)
    plt.xlabel("Fitness")
    plt.ylabel("Frecuencia")
    plt.title("Histograma del Fitness en la Última Generación")
    
    #mostrar gráficos
    plt.show()

# ------------------------------
# Función principal
# ------------------------------
def main():
    # Cargar configuración desde JSON
    config = load_config("config.json")
    # Inicializa el algoritmo genético
    ga = GeneticAlgorithm(config)
    fitness_history_avg, fitness_history_max, fitness_distribution = ga.run()
    
    # Selecciona los 3 agentes con mayor fitness de la última generación para visualizarlos
    final_fitness = [ga.evaluate_agent(agent) for agent in ga.population]
    sorted_indices = np.argsort(final_fitness)[::-1]
    top_agents = [ga.population[i] for i in sorted_indices[:3]]
    visualize_top_agents(top_agents, config["board_size"])
    
    # Genera los gráficos de evolución y distribución del fitness
    plot_results(fitness_history_avg, fitness_history_max, fitness_distribution)

if __name__ == "__main__":
    main()
