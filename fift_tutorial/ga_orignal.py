import time

import numpy as np


class IHMGA:
    def __str__(self):
        return "IHMGA"

    def __init__(self, pop_size, dna_size, bound, fitness_func, feature_importance, fi_method='shap', fi_threshold_type='percentile', crossover_rate=0.5, mutation_rate=0.1, mutation_type='ihm'):
        self.mutation_type = mutation_type
        self.pop_size = pop_size
        self.dna_size = dna_size
        self.bound = np.array(bound)
        self.fitness_func = fitness_func
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.feature_importance = feature_importance
        self.fi_method = fi_method
        self.alpha = 0.1  # mutation scaling factor.
        
        if fi_threshold_type == 'mean':
            self.importance_threshold = np.mean(feature_importance)
        elif fi_threshold_type == 'std':
            self.importance_threshold = np.std(feature_importance)
        elif fi_threshold_type == '25_percentile':
            self.importance_threshold = np.percentile(feature_importance, 25)
        elif fi_threshold_type == '75_percentile':
            self.importance_threshold = np.percentile(feature_importance, 75)
        elif fi_threshold_type == '50_percentile':
            self.importance_threshold = np.percentile(feature_importance, 50)
        elif fi_threshold_type == 'simple':
            self.importance_threshold = 0.05  # threshold for standard feature importance
        else:
            raise ValueError("fi_threshold_type must be one of ['mean', 'std', 'simple', 'percentile']")
        
        self.generation = 0
        self.tournament_size = max(5, int(self.pop_size * 0.05))

        self.generate_population()

    def generate_population(self):
        self.population = np.random.uniform(
            low=self.bound[:, 0],
            high=self.bound[:, 1],
            size=(self.pop_size, self.dna_size)
        )

    def select(self):
        fitness = self.fitness_func(self.population)
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
        new_population = []
        indices = np.arange(self.pop_size)
        np.random.shuffle(indices)   # random pairing

        for i in range(0, self.pop_size, 2):
            p1 = self.population[indices[i]].copy()
            p2 = self.population[indices[i+1]].copy()

            # children initially copies of parents
            c1, c2 = p1.copy(), p2.copy()

            # uniform crossover: loop gene by gene
            for j in range(self.dna_size):
                if np.random.rand() < self.crossover_rate:
                    c1[j], c2[j] = c2[j], c1[j]

            new_population.append(c1)
            new_population.append(c2)

        self.population = np.array(new_population)

    # inverse important mutation (IIM)
    def iim(self, i, j):
        """We introduce a per-gene mutation magnitude scaled by α(1−Ij), where Ij is the 
        normalized feature importance of gene j. This design is inspired by adaptive mutation 
        strategies in evolutionary computation and knowledge-guided operator designs.
        """
        # Scale inversely with feature importance
        importance = self.feature_importance[j]
        # smaller conservative mutation for important features
        mutation_scale = (1 - importance)*self.alpha  
        mutation_range = (self.bound[j, 1] - self.bound[j, 0]) * mutation_scale
        self.population[i, j] += np.random.uniform(-mutation_range, mutation_range)
    
    # Direct importance mutation
    def dim(self, i, j):
        # Scale directly with feature importance
        mutation_scale = self.feature_importance[j]*self.alpha  
        mutation_range = (self.bound[j, 1] - self.bound[j, 0]) * mutation_scale
        self.population[i, j] += np.random.uniform(-mutation_range, mutation_range)

    # Uniform Reset Mutation
    def uniform_mutation(self, i, j):
        """Uniform reset mutation replaces the gene value with a new random value
        sampled uniformly within the feature's valid bounds [low, high]."""
        self.population[i, j] = np.random.uniform(self.bound[j, 0], self.bound[j, 1])


    # Importance-Guided Hybrid Mutation
    def ihm(self, i, j):
        """We adapt mutation magnitude per gene inversely with an empirically derived 
        feature-importance score. We modulate mutation magnitude per gene inversely with a feature-importance score. This is motivated
        by two lines of work: adaptive/gene-level mutation strategies in evolutionary computation that tune perturbation
        intensity to balance exploration and exploitation, and knowledge-guided evolutionary methods that bias 
        genetic operators using prior information about features. Conceptually this is analogous to biological 
        systems where functionally important regions are subject to lower mutation rates. By reducing mutation 
        on high-importance features we preserve promising structure while allowing more aggressive exploration 
        in less important dimensions.
        Mutation magnitude for gene j was scaled inversely with its normalized importance Ij as: Δj=α(1−Ij)Rj 
        where Rj is the feature range and α is a damping hyperparameter. For features with Ij above a threshold 
        we applied perturbation noise U(−Δj,Δj); otherwise we applied a broader reset. 
        This choice follows adaptive gene-level mutation strategies in EC literature and knowledge-guided operator 
        designs. REF: https://www.mdpi.com/2076-3417/14/11/4501; https://www.sciencedirect.com/science/article/pii/S1568494622003052.
        """

        importance = self.feature_importance[j]
        if importance > self.importance_threshold:
            self.iim(i, j)
        else:
            # less important features get full reset (explore)
            self.uniform_mutation(i, j)

    def mutate(self):
        
        for i in range(self.pop_size):
            for j in range(self.dna_size):
                if np.random.rand() < self.mutation_rate:
                    if self.mutation_type == "iim":
                        self.iim(i, j)
                    elif self.mutation_type == "ihm":
                        self.ihm(i, j)
                    elif self.mutation_type == "dim":
                        self.dim(i, j)
                    else:
                        # uniform random mutation
                        self.uniform_mutation(i, j)
                    # enforce bounds
                    self.population[i, j] = np.clip(self.population[i, j], self.bound[j, 0], self.bound[j, 1])
       

    def evolve(self):
        self.select()
        self.crossover()
        self.mutate()