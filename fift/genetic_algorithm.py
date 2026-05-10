import time
import numpy as np


class GA:
    def __str__(self):
        return "GA"

    def __init__(self, pop_size, dna_size, bound, fitness_func, feature_importance, mutation_rate=0.095):
        self.pop_size = pop_size
        self.dna_size = dna_size
        self.bound = np.array(bound)
        self.fitness_func = fitness_func
        self.mutation_rate = mutation_rate

        self.feature_importance = np.array(feature_importance, dtype=float)

        self.tournament_size = max(5, int(self.pop_size * 0.05))

        self.generate_population()

    def generate_population(self):
        self.population = np.random.uniform(
            low=self.bound[:, 0],
            high=self.bound[:, 1],
            size=(self.pop_size, self.dna_size)
        )

    def evaluate_fitness(self):
        # fitness = np.array([self.fitness_func(inp) for inp in self.population]) + 1e-3 # avoid division by zero
        fitness = self.fitness_func(self.population) + 1e-3 # avoid division by zero
        return fitness

    def select(self, fitness):
        selection_probs = fitness / np.sum(fitness)

        selected = []
        for _ in range(self.pop_size):
            tournament = np.random.choice(
                np.arange(self.pop_size),
                self.tournament_size,
                replace=False,
                p=selection_probs
            )
            winner = tournament[np.argmax(fitness[tournament])]
            selected.append(self.population[winner])

        self.population = np.vstack(selected)

    def crossover(self):
        for i in range(0, self.pop_size, 2):
            partner_idx = np.random.randint(0, self.pop_size)
            for j in range(self.dna_size):
                if self.population[i, j] != self.population[partner_idx, j]:
                    if np.random.rand() < self.feature_importance[j]:
                        self.population[i, j], self.population[partner_idx, j] = \
                            self.population[partner_idx, j], self.population[i, j]

    def mutate(self):
        for i in range(self.pop_size):
            for j in range(self.dna_size):
                if np.random.rand() < self.mutation_rate:
                    mutation_range = (self.bound[j, 1] - self.bound[j, 0])
                    self.population[i, j] += np.random.uniform(-mutation_range, mutation_range) * self.feature_importance[j]

                    self.population[i, j] = np.clip(self.population[i, j], self.bound[j, 0], self.bound[j, 1])


    def run(self):
        fitness = self.evaluate_fitness()
        self.select(fitness)
        self.crossover()
        self.mutate()

