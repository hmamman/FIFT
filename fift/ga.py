import time
import numpy as np


class GA:
    def __str__(self):
        return "GA"

    def __init__(self, pop_size, dna_size, bound, fitness_func, feature_importance, cross_rate=0.8, mutation_rate=0.2,
                 fi_percentile=25): # keeps the top 75% of features
        self.pop_size = pop_size
        self.dna_size = dna_size
        self.bound = np.array(bound)
        self.fitness_func = fitness_func
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate

        # Normalize FI to [0,1]
        self.feature_importance = np.array(feature_importance, dtype=float)
        # self.feature_importance = (self.feature_importance - self.feature_importance.min()) / (
        #             self.feature_importance.ptp() + 1e-12)
        # normalized =feature_importance/np.sum(feature_importance)
        #  = normalized
        
        # print(self.feature_importance, np.mean(self.feature_importance))
        # print(normalized, np.mean(normalized))
        # exit()

        # Dynamic threshold based on percentile
        self.importance_threshold = 0.05 # np.percentile(self.feature_importance, fi_percentile)

        self.generation = 0
        self.tournament_size = max(5, int(self.pop_size * 0.05))

        self.generate_population()

    def generate_population(self):
        self.population = np.random.uniform(
            low=self.bound[:, 0],
            high=self.bound[:, 1],
            size=(self.pop_size, self.dna_size)
        )

    def get_fitness(self):
        fitness = np.array([self.fitness_func(fit) for fit in self.population]) + 1e-3
        # fitness = self.fitness_func(self.population) + 1e-3
        return fitness

    def select(self):
        fitness = self.get_fitness()
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
            if np.random.rand() < self.cross_rate:
                partner_idx = np.random.randint(0, self.pop_size)
                for j in range(self.dna_size):
                    if self.feature_importance[j] > self.importance_threshold:
                        # Higher FI → higher crossover chance
                        if np.random.rand() < self.feature_importance[j]:
                            self.population[i, j], self.population[partner_idx, j] = \
                                self.population[partner_idx, j], self.population[i, j]

    def mutate(self):
        for i in range(self.pop_size):
            for j in range(self.dna_size):
                if np.random.rand() < self.mutation_rate * (1 + self.feature_importance[j]):
                    if self.feature_importance[j] > self.importance_threshold:
                        mutation_range = (self.bound[j, 1] - self.bound[j, 0]) * self.feature_importance[j]
                        self.population[i, j] += np.random.uniform(-mutation_range, mutation_range)
                    else:
                        self.population[i, j] = np.random.uniform(self.bound[j, 0], self.bound[j, 1])

                    self.population[i, j] = np.clip(self.population[i, j], self.bound[j, 0], self.bound[j, 1])

    def run(self):
        self.select()
        self.crossover()
        self.mutate()
        self.generation += 1

