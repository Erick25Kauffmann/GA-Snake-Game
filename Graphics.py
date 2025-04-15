import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool

# ===============================
# Función para graficar resultados
# ===============================
def plot_results(fitness_history_avg, fitness_history_max, apple_count_history_avg, apple_count_history_max, fitness_distribution):
    generations = list(range(len(fitness_history_avg)))
    plt.figure("Evolución del Fitness")
    plt.plot(generations, fitness_history_avg, label="Fitness Promedio")
    plt.plot(generations, fitness_history_max, label="Fitness Máximo")
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.title("Evolución del Fitness")
    plt.legend()
    plt.grid(True)
    
    plt.figure("Evolución del Apple Count")
    plt.plot(generations, apple_count_history_avg, label="Apple Count Promedio")
    plt.plot(generations, apple_count_history_max, label="Apple Count Máximo")
    plt.xlabel("Generación")
    plt.ylabel("Manzanas")
    plt.title("Evolución del Apple Count")
    plt.legend()
    plt.grid(True)
    
    plt.figure("Distribución del Fitness (Última Generación)")
    plt.bar(range(len(fitness_distribution[-1])), fitness_distribution[-1])
    plt.xlabel("Individuo")
    plt.ylabel("Fitness")
    plt.title("Distribución del Fitness en la Última Generación")
    
    df = pd.DataFrame(fitness_distribution).T
    plt.figure("Boxplot del Fitness por Generación")
    df.boxplot()
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.title("Boxplot del Fitness por Generación")
    
    plt.figure("Histograma del Fitness (Última Generación)")
    plt.hist(fitness_distribution[-1], bins=10)
    plt.xlabel("Fitness")
    plt.ylabel("Frecuencia")
    plt.title("Histograma del Fitness en la Última Generación")
    
    plt.show()