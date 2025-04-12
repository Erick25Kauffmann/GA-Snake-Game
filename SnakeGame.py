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
    def __init__(self, decision_table=None, generation=0):
        # La tabla de decisión se inicializa como una lista vacía (si no se pasa una)
        # Cada fila será: (state, decision, steps_count, apples_count)
        if decision_table is None:
            self.decision_table = []
        else:
            self.decision_table = decision_table
        self.generation = generation
        self.fitness = 0
        self.final_apples = 0

    def decide(self, game: SnakeGame):
        current_state = game.board.copy()
        # Intentamos buscar en la tabla heredada un estado similar al actual.
        # Usamos un umbral para determinar similitud (puedes ajustar este valor).
        threshold = 5  # Número máximo de celdas diferentes para considerar similares
        best_diff = float('inf')
        guided_decision = None
        for row in self.decision_table:
            stored_state, stored_decision, stored_steps, stored_apples = row
            # Calculamos la cantidad de celdas diferentes (Hamming distance)
            diff = np.sum(current_state != stored_state)
            if diff < best_diff and diff < threshold:
                best_diff = diff
                guided_decision = stored_decision
        if guided_decision is not None:
            decision = guided_decision
        else:
            decision = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
        # Independientemente del método de elección, registramos la información del paso
        steps_count = game.steps
        apples_count = game.food
        self.decision_table.append((current_state, decision, steps_count, apples_count))
        return decision

    def mutate(self, mutation_probability):
        # Para cada fila, con la probabilidad indicada, se cambia la decision
        for i in range(len(self.decision_table)):
            if random.random() < mutation_probability:
                state, _, steps_count, apples_count = self.decision_table[i]
                new_decision = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
                self.decision_table[i] = (state, new_decision, steps_count, apples_count)

    @staticmethod
    def crossover(parent1, parent2):
        table1 = parent1.decision_table
        table2 = parent2.decision_table
        min_length = min(len(table1), len(table2))
        # Si alguna tabla está vacía, se copia la no vacía (o se queda vacía)
        if min_length == 0:
            new_table = copy.deepcopy(table1 if len(table1) > 0 else table2)
            return DecisionTableAgent(new_table)
        # Elegir un punto de cruce (en el rango 1 a min_length-1)
        crossover_point = random.randint(1, min_length - 1)
        new_table = table1[:crossover_point] + table2[crossover_point:]
        return DecisionTableAgent(new_table)


# ------------------------------
# Clase para el Algoritmo Genético
# ------------------------------
def manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


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
        self.apple_count_history_avg = []
        self.apple_count_history_max = []
        self.fitness_distribution = []  # Lista con los fitness de cada generación

        self.best_agents_by_gen = []
        
    
    def evaluate_agent(self, agent, max_steps=2000):
        game = SnakeGame(self.board_size)
        stats = []
        # Aquí mantenemos la tabla heredada (o, si prefieres, puedes reiniciarla)
        # agent.decision_table = []  # O dejarla para aprovechar el conocimiento heredado
        movement_penalty = 0
        penalty_coefficient = 10  # Este coeficiente lo puedes ajustar
        # Calculamos la distancia inicial desde la cabeza al apple
        previous_distance = manhattan_distance(game.snake[0], game.apple_position)
        
        # Simula la partida
        while game.alive and game.steps < max_steps:
            action = agent.decide(game)
            old_apple_position = game.apple_position  # Para saber si se cambia al comer
            game.move(action)
            
            if game.alive:
                new_distance = manhattan_distance(game.snake[0], game.apple_position)
                # Si la manzana no se comió (es decir, no cambió de posición) y la distancia aumentó,
                # acumulamos penalización:
                if old_apple_position == game.apple_position and new_distance > previous_distance:
                    movement_penalty += (new_distance - previous_distance) * penalty_coefficient
                previous_distance = new_distance
        # Penalización por “muerte temprana”
        premature_death = 1 if game.food <= agent.generation else 0
        final_fitness = (game.food * 200) - (game.steps * 3) - (premature_death * 50) - movement_penalty
        agent.fitness = final_fitness
        agent.final_apples = game.food
        
        if len(agent.decision_table) > 0:
            last_state, last_decision, _, _ = agent.decision_table[-1]
            agent.decision_table[-1] = (last_state.copy(), last_decision, game.steps, game.food)
        else:
            agent.decision_table.append((game.board.copy(), None, game.steps, game.food))
        stats.append(final_fitness)
        stats.append(agent.final_apples)
        return stats


    
    def evaluate_population(self):
        population_stats = []
        fitness_values = []
        final_apples_values = []
        for agent in self.population:
            #fitness = self.evaluate_agent(agent)
            agent_stats = self.evaluate_agent(agent)
            fitness = agent_stats[0]
            final_apples = agent_stats[1]
            fitness_values.append(fitness)
            final_apples_values.append(final_apples)
        population_stats.append(fitness_values)
        population_stats.append(final_apples_values)
        #return fitness_values
        return population_stats
    
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
            #fitness_values = self.evaluate_population()
            population_stats = self.evaluate_population()
            fitness_values = population_stats[0]
            final_apples_values = population_stats[1]
            avg_fit = np.mean(fitness_values)
            max_fit = np.max(fitness_values)
            avg_apples = np.mean(final_apples_values)
            max_apples = np.max(final_apples_values)
            self.fitness_history_avg.append(avg_fit)
            self.fitness_history_max.append(max_fit)
            self.apple_count_history_avg.append(avg_apples)
            self.apple_count_history_max.append(max_apples)
            self.fitness_distribution.append(fitness_values)
            
            print(f"Generación {generation}: fitness promedio = {avg_fit:.2f}, fitness máximo = {max_fit:.2f}, promedio manzanas = {avg_apples:.2f}, máximo manzanas = {max_apples:.2f}")

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
        return self.fitness_history_avg, self.fitness_history_max, self.apple_count_history_avg, self.apple_count_history_max, self.fitness_distribution

# ------------------------------
# Visualización en Pygame de los Top 3 Agentes
# ------------------------------
def visualize_top_agents(agents, board_size, cell_size=20):
    pygame.init()
    # Calculamos el ancho necesario: un tablero por agente
    width = board_size * cell_size * len(agents)
    height = board_size * cell_size + 50  # Espacio adicional para el texto
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Visualización de Top Agentes")
    font = pygame.font.SysFont(None, 24)
    clock = pygame.time.Clock()
    
    # Usamos el número máximo de registros (filas) que tengan los agentes en su decision_table.
    max_length = max(len(agent.decision_table) for agent in agents)
    
    running = True
    step_index = 0
    
    while running and step_index < max_length:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.fill((0, 0, 0))
        
        for i, agent in enumerate(agents):
            # Usamos agent.decision_table (ya que allí se guardan 4 elementos por fila)
            if step_index < len(agent.decision_table):
                board, _, steps, apples_count = agent.decision_table[step_index]
            else:
                board, _, steps, apples_count = agent.decision_table[-1]
            
            # Dibujar el tablero
            for x in range(board_size):
                for y in range(board_size):
                    cell_value = board[y, x]
                    rect = pygame.Rect(i * board_size * cell_size + x * cell_size,
                                         y * cell_size, cell_size, cell_size)
                    if cell_value == EMPTY:
                        color = (40, 40, 40)
                    elif cell_value == SNAKE_HEAD:
                        color = (0, 255, 0)
                    elif cell_value == SNAKE_BODY:
                        color = (0, 200, 0)
                    elif cell_value == APPLE:
                        color = (255, 0, 0)
                    pygame.draw.rect(screen, color, rect)
            
            # Dibujar un borde blanco alrededor de cada tablero
            border_rect = pygame.Rect(i * board_size * cell_size,
                                      0, board_size * cell_size, board_size * cell_size)
            pygame.draw.rect(screen, (255, 255, 255), border_rect, width=1)
            
            # Mostrar dinámicamente el fitness y el número de manzanas consumidas (apple count)
            text_fit = font.render(f"Steps: {steps}", True, (255, 255, 255))
            text_apples = font.render(f"Apples: {apples_count}", True, (255, 255, 255))
            screen.blit(text_fit, (i * board_size * cell_size, board_size * cell_size))
            screen.blit(text_apples, (i * board_size * cell_size, board_size * cell_size + 20))
        
        pygame.display.flip()
        clock.tick(8)  # Controla la velocidad de la animación
        step_index += 1
    
    # Mantener la ventana abierta hasta cerrar manualmente
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
def plot_results(fitness_history_avg, fitness_history_max, apple_count_history_avg, apple_count_history_max, fitness_distribution):
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
    
    # Gráfico de líneas: evolución del Apple Count promedio y máximo
    plt.figure("Evolución del Apple Count")
    plt.plot(generations, apple_count_history_avg, label="Apple Count Promedio")
    plt.plot(generations, apple_count_history_max, label="Apple Count Máximo")
    plt.xlabel("Generación")
    plt.ylabel("Cantidad de Manzanas")
    plt.title("Evolución del Apple Count")
    plt.legend()
    plt.grid(True)
    
    # Gráfico de barras: distribución del fitness en la última generación
    plt.figure("Distribución del Fitness (Última Generación)")
    plt.bar(range(len(fitness_distribution[-1])), fitness_distribution[-1])
    plt.xlabel("Individuo")
    plt.ylabel("Fitness")
    plt.title("Distribución del Fitness en la Última Generación")
    
    # Boxplot para visualizar la distribución por generación
    df = pd.DataFrame(fitness_distribution).T
    plt.figure("Boxplot del Fitness por Generación")
    df.boxplot()
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.title("Boxplot del Fitness por Generación")
    
    # Histograma del fitness en la última generación
    plt.figure("Histograma del Fitness (Última Generación)")
    plt.hist(fitness_distribution[-1], bins=10)
    plt.xlabel("Fitness")
    plt.ylabel("Frecuencia")
    plt.title("Histograma del Fitness en la Última Generación")
    
    plt.show()


# ------------------------------
# Función principal
# ------------------------------
def main():
    # Cargar configuración desde JSON
    config = load_config("config.json")
    # Inicializa el algoritmo genético
    ga = GeneticAlgorithm(config)
    fitness_history_avg, fitness_history_max, apple_count_avg, apple_count_max, fitness_distribution = ga.run()
    
    # Selecciona los 3 agentes con mayor fitness de la última generación para visualizarlos
    final_fitness = [agent.fitness for agent in ga.population]
    sorted_indices = np.argsort(final_fitness)[::-1]
    top_agents = [ga.population[i] for i in sorted_indices[:3]]
    visualize_top_agents(top_agents, config["board_size"])
    
    # Genera los gráficos de evolución y distribución del fitness
    plot_results(fitness_history_avg, fitness_history_max, apple_count_avg, apple_count_max, fitness_distribution)

if __name__ == "__main__":
    main()
