from poa import evaluate, search_strategy, objective_function, initialize
from variables import *
import numpy as np

pelican_positions, pelican_velocities, fish_positions = initialize(n_variables, n_pelicans, n_fishes)

best_position = None
best_fitness = np.inf

# Iterasi POA
for i in range(n_iterations):
    for j in range(n_pelicans):
        new_position, velocity = search_strategy(pelican_positions[j], pelican_positions, fish_positions, p_greediness, w, lower_bound, upper_bound)
        old_fitness = evaluate(pelican_positions[j], objective_function)
        new_fitness = evaluate(new_position, objective_function)
        if new_fitness < old_fitness:
            pelican_positions[j] = new_position
            pelican_velocities[j] = velocity
            if new_fitness < best_fitness:
                best_position = new_position
                best_fitness = new_fitness
    for j in range(n_fishes):
        fish_positions[j] = np.random.uniform(-10, 10, n_variables)
    if (i+1) % 10 == 0:
        print("Iterasi {}: Posisi Terbaik = {}, Fitness = {}".format(i+1, best_position, best_fitness))