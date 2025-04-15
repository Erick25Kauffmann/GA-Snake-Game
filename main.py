import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool

from Functions import *
from GeneticAlgorithm import *
from Visualization import *
from Graphics import *

# ===============================
# Función principal
# ===============================
def main():
    config = load_config("config.json")
    ga = GeneticAlgorithm(config)
    (fitness_history_avg, fitness_history_max, 
     apple_count_history_avg, apple_count_history_max, 
     fitness_distribution) = ga.run()
    
    final_fitness = [agent.fitness for agent in ga.population]
    sorted_indices = np.argsort(final_fitness)[::-1]
    top_agents = [ga.population[i] for i in sorted_indices[:config["top_visualization"]]]
    
    runs = []
    for agent in top_agents:
        run = run_single_game(agent, config["board_size"])
        runs.append(run)
    
    visualize_runs(runs, config["board_size"])
    plot_results(fitness_history_avg, fitness_history_max, 
                 apple_count_history_avg, apple_count_history_max, fitness_distribution)

if __name__ == "__main__":
    main()
