import numpy as np

def initialize(n_variables, n_pelicans, n_fishes):
    pelican_positions = np.zeros((n_pelicans, n_variables))
    pelican_velocities = np.zeros((n_pelicans, n_variables))
    fish_positions = np.zeros((n_fishes, n_variables))
    for i in range(n_pelicans):
        pelican_positions[i] = np.random.uniform(-10, 10, n_variables)
        pelican_velocities[i] = np.random.uniform(-1, 1, n_variables)
    for i in range(n_fishes):
        fish_positions[i] = np.random.uniform(-10, 10, n_variables)
    return pelican_positions, pelican_velocities, fish_positions