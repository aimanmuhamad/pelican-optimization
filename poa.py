import numpy as np
from variables import *

def objective_function(x):
    return x**2

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

def search_direct(pelican_position, fish_positions):
    n_fishes = len(fish_positions)
    distances = np.zeros(n_fishes)
    for i in range(n_fishes):
        distances[i] = np.linalg.norm(pelican_position - fish_positions[i])
    return fish_positions[np.argmin(distances)]

def search_indirect(pelican_position, pelican_positions, fish_positions, w, lower_bound, upper_bound):
    other_pelican_position = pelican_positions[np.random.randint(len(pelican_positions))]
    displacement_vector = other_pelican_position - pelican_position
    velocity = w * displacement_vector
    new_position = np.clip(pelican_position + velocity, lower_bound, upper_bound)
    return new_position, velocity

def search_patrol(pelican_position, pelican_positions, fish_positions):
    n_pelicans = len(pelican_positions)
    distances = np.zeros(n_pelicans)
    for i in range(n_pelicans):
        if not np.array_equal(pelican_position, pelican_positions[i]):
            distances[i] = np.linalg.norm(pelican_position - pelican_positions[i])
        else:
            distances[i] = np.inf
    closest_pelican_position = pelican_positions[np.argmin(distances)]
    closest_fish_position = search_direct(closest_pelican_position, fish_positions)
    return 2 * closest_fish_position - pelican_position

def search_random(pelican_position, lower_bound, upper_bound, w):
    pelican_positions, pelican_velocities, fish_positions = initialize(n_variables, n_pelicans, n_fishes)
    r1 = np.random.uniform(0, 1, len(pelican_position))
    r2 = np.random.uniform(0, 1, len(pelican_position))
    velocity = w * r1 - (pelican_position - search_patrol(pelican_position, pelican_positions, fish_positions)) * r2
    new_position = pelican_position + velocity
    new_position = np.maximum(new_position, lower_bound)
    new_position = np.minimum(new_position, upper_bound)
    return new_position, velocity

def search_strategy(pelican_position, pelican_positions, fish_positions, p_greediness, w, lower_bound, upper_bound):
    if np.random.uniform(0, 1) < p_greediness:
        return search_direct(pelican_position, fish_positions), None
    else:
        return search_indirect(pelican_position, pelican_positions, fish_positions, w, lower_bound, upper_bound)
        
def evaluate(position, objective_function):
    return objective_function(position)
