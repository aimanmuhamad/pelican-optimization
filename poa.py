import numpy as np
from variables import *

class PelicanOptimization():
    def __init__(self, w, n_variables, n_pelicans, n_fishes, pelican_position,
                 fish_position, lower_bound, upper_bound, p_greediness):
        self.w = w
        self.n_variables = n_variables
        self.n_pelicans = n_pelicans
        self.n_fishes = n_fishes
        self.pelican_position = pelican_position
        self.fish_position = fish_position
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.p_greediness = p_greediness
        
    def initialize(self, n_variables, n_pelicans, n_fishes):
        pelican_positions = np.zeros((n_pelicans, n_variables))
        pelican_velocities = np.zeros((n_pelicans, n_variables))
        fish_position = np.zeros((n_fishes, n_variables))
        for i in range(n_pelicans):
            pelican_positions[i] = np.random.uniform(-10, 10, n_variables)
            pelican_velocities[i] = np.random.uniform(-1, 1, n_variables)
        for i in range(n_fishes):
            fish_position[i] = np.random.uniform(-10, 10, n_variables)
        return pelican_positions, pelican_velocities, fish_position

    def search_direct(self, pelican_position, fish_position):
        n_fishes = len(fish_position)
        distances = np.zeros(n_fishes)
        for i in range(n_fishes):
            distances[i] = np.linalg.norm(pelican_position - fish_position[i])
        return fish_position[np.argmin(distances)]

    def search_indirect(self, pelican_position, pelican_positions, fish_position, w, lower_bound, upper_bound):
        other_pelican_position = pelican_positions[np.random.randint(len(pelican_positions))]
        displacement_vector = other_pelican_position - pelican_position
        velocity = w * displacement_vector
        new_position = np.clip(pelican_position + velocity, lower_bound, upper_bound)
        return new_position, velocity

    def search_patrol(self, pelican_position, pelican_positions, fish_position):
        n_pelicans = len(pelican_position)
        distances = np.zeros(n_pelicans)
        for i in range(n_pelicans):
            if not np.array_equal(pelican_position, pelican_positions[i]):
                distances[i] = np.linalg.norm(pelican_position - pelican_positions[i])
            else:
                distances[i] = np.inf
        closest_pelican_position = pelican_positions[np.argmin(distances)]
        closest_fish_position = self.search_direct(closest_pelican_position, fish_position)
        return 2 * closest_fish_position - pelican_position

    def search_random(self, pelican_position, lower_bound, upper_bound, w):
        pelican_positions, pelican_velocities, fish_position = self.initialize(n_variables, n_pelicans, n_fishes)
        r1 = np.random.uniform(0, 1, len(pelican_position))
        r2 = np.random.uniform(0, 1, len(pelican_position))
        velocity = w * r1 - (pelican_position - self.search_patrol(pelican_position, pelican_positions, fish_position)) * r2
        new_position = pelican_position + velocity
        new_position = np.maximum(new_position, lower_bound)
        new_position = np.minimum(new_position, upper_bound)
        return new_position, velocity

    def search_strategy(self, pelican_position, pelican_positions, fish_position, p_greediness, w, lower_bound, upper_bound):
        if np.random.uniform(0, 1) < p_greediness:
            return self.search_direct(pelican_position, fish_position), None
        else:
            return self.search_indirect(pelican_position, pelican_positions, fish_position, w, lower_bound, upper_bound)
            
    def evaluate(self, position, objective_function):
        return objective_function(position)

    def objective_function(self, x):
        return x**2
