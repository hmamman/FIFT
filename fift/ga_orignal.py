import time

import numpy as np


class GA:
    def __str__(self):
        return "GA"

    def __init__(self, pop_size, dna_size, bound, fitness_func, feature_importance, fi_method='shap', fi_threshold_type='percentile', crossover_rate=0.5, mutation_rate=0.1):
        self.pop_size = pop_size
        self.dna_size = dna_size
        self.bound = np.array(bound)
        self.fitness_func = fitness_func
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.feature_importance = feature_importance
        self.fi_method = fi_method
        self.alpha = 0.1  # mutation scaling factor. 
        # Ablation: try different α values (e.g. 0.01, 0.05, 0.1, 0.2) to see how sensitive performance is to α.
        """In our experiments, we normalized feature importance to lie in [0,1]. We considered a feature ‘important’ 
        if its importance value exceeded the mean of all importance values as in Tsai and Sung (2020) https://doi.org/10.1016/j.knosys.2020.106097. 
        These thresholds are consistent with 
        feature selection ensemble studies (e.g. Alzheimer’s biomarker ensemble FS-MSI using 75th quartile values) 
        and IG-based methods that use statistical thresholds (e.g. standard deviation) for filtering weak features. 
        We also performed sensitivity analysis across threshold choices (median, 75th percentile, mean+½ sd) 
        to assess robustness of the GA with respect to this parameter."""
        
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
        # self.importance_threshold = np.mean(feature_importance)
        # we can equally use median or 75th percentile as threshold
        # self.importance_threshold = np.median(feature_importance)
        # self.importance_threshold = np.percentile(feature_importance, 75)
        # Another method is to use the standard deviation above the mean as in:
        # self.importance_threshold = np.std(feature_importance)

        # k = 0.05  # threshold scaling factor, hyperparameter to tune
        # self.importance_threshold = np.mean(feature_importance) # + k * np.std(feature_importance)
        # # use log2 scale for thresholding
        # # self.importance_threshold = np.log2(np.sum(feature_importance))
        # print(f"Feature importance threshold set to {self.importance_threshold:.4f} (mean + {k}*std)")
        # print(f"Feature importance: {[round(fi,5) for fi in self.feature_importance]}")
        # exit()

        
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
        """Uniform crossover applied in-place to the whole population.
        We use a uniform crossover operator with probability 0.5 applied at the gene level. 
        Additionally, when using feature importance guidance, crossover is restricted to features 
        with importance exceeding a predefined threshold."""
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
                    # if self.fi_method == "uniform":
                    #     c1[j], c2[j] = c2[j], c1[j]
                    # elif self.feature_importance[j] > self.importance_threshold:
                    #     # “We apply a uniform crossover operator (p=0.5 per gene) restricted to features exceeding a defined importance threshold.”
                    #     c1[j], c2[j] = c2[j], c1[j]

            new_population.append(c1)
            new_population.append(c2)

        self.population = np.array(new_population)

    
    def mutate(self):
        """We adapt mutation magnitude per gene inversely with an empirically derived 
        feature-importance score. This design is inspired by biological evidence that functionally 
        critical genes exhibit lower mutation rates (e.g. in Arabidopsis thaliana) and by evolutionary 
        computation techniques that balance exploitation and exploration by reducing perturbation in promising 
        regions. By mutating “important” features less, we preserve structural stability in those dimensions, 
        while allowing more aggressive exploration in features of low importance. This enables the GA to 
        refine solutions without losing the gains in key features. REF: https://pubmed.ncbi.nlm.nih.gov/35022609/"""
        """We modulate mutation magnitude per gene inversely with a feature-importance score. This is motivated 
        by two lines of work: adaptive/gene-level mutation strategies in evolutionary computation that tune perturbation
        intensity to balance exploration and exploitation, and knowledge-guided evolutionary methods that bias 
        genetic operators using prior information about features. Conceptually this is analogous to biological 
        systems where functionally important regions are subject to lower mutation rates. By reducing mutation 
        on high-importance features we preserve promising structure while allowing more aggressive exploration 
        in less important dimensions."""
        for i in range(self.pop_size):
            for j in range(self.dna_size):
                if np.random.rand() < self.mutation_rate:
                    if self.fi_method != "uniform":
                        # Scale inversely with feature importance
                        importance = self.feature_importance[j]
                        if importance > self.importance_threshold:
                            """We introduce a per-gene mutation magnitude scaled by α(1−Ij), where Ij is the 
                            normalized feature importance of gene j. This design is inspired by adaptive mutation 
                            strategies in evolutionary computation and knowledge-guided operator designs. 
                            The scaling factor α α serves as a hyperparameter to limit mutation step sizes 
                            and avoid overly large perturbations even for features deemed ‘unimportant’. 
                            We evaluate through experiments over several α values to test sensitivity.
                            """
                            # smaller conservative mutation for important features
                            mutation_scale = (1 - importance)*self.alpha  
                            mutation_range = (self.bound[j, 1] - self.bound[j, 0]) * mutation_scale
                            self.population[i, j] += np.random.uniform(-mutation_range, mutation_range)
                        else:
                            # less important features get full reset (explore)
                            self.population[i, j] = np.random.uniform(self.bound[j, 0], self.bound[j, 1])
                    else:
                        # standard GA: random reset mutation
                        self.population[i, j] = np.random.uniform(self.bound[j, 0], self.bound[j, 1])

                    # enforce bounds
                    self.population[i, j] = np.clip(self.population[i, j], self.bound[j, 0], self.bound[j, 1])
        """
        Mutation magnitude for gene j was scaled inversely with its normalized importance Ij as: Δj=α(1−Ij)Rj 
        where Rj is the feature range and α is a damping hyperparameter. For features with Ij above a threshold 
        we applied perturbation noise U(−Δj,Δj); otherwise we applied a broader reset. 
        This choice follows adaptive gene-level mutation strategies in EC literature and knowledge-guided operator 
        designs. REF: https://www.mdpi.com/2076-3417/14/11/4501; https://www.sciencedirect.com/science/article/pii/S1568494622003052.
        """

    def evolve(self):
        self.select()
        self.crossover()
        self.mutate()


import time
import numpy as np


class GA_Optimized:
    def __str__(self):
        return "GA_Optimized"

    def __init__(self, pop_size, dna_size, bound, fitness_func, feature_importance, fi_method='shap',
                 fi_threshold_type='percentile', crossover_rate=0.5, mutation_rate=0.1):

        # Ensure pop_size is even for crossover
        if pop_size % 2 != 0:
            pop_size += 1
            print(f"Warning: pop_size must be even. Setting to {pop_size}.")

        self.pop_size = pop_size
        self.dna_size = dna_size
        self.bound = np.array(bound)
        self.fitness_func = fitness_func
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.feature_importance = np.array(feature_importance)  # Ensure it's a numpy array
        self.fi_method = fi_method
        self.alpha = 0.1

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
            self.importance_threshold = 0.05
        else:
            raise ValueError(
                "fi_threshold_type must be one of ['mean', 'std', 'simple', '25_percentile', '50_percentile', '75_percentile']")

        self.generation = 0
        self.tournament_size = max(5, int(self.pop_size * 0.05))

        # --- Pre-calculate arrays for vectorization ---

        # Bounds (shape: (dna_size,))
        self.low_bound = self.bound[:, 0]
        self.high_bound = self.bound[:, 1]
        self.gene_range = self.high_bound - self.low_bound

        # Mask for important features (shape: (dna_size,))
        # This will broadcast to (pop_size, dna_size) when needed
        self.is_important_mask = self.feature_importance > self.importance_threshold

        # Mutation ranges for "important" features (shape: (dna_size,))
        self.mutation_scale = (1 - self.feature_importance) * self.alpha
        self.mutation_range = self.gene_range * self.mutation_scale

        # --- End Pre-calculation ---

        self.generate_population()

    def generate_population(self):
        # This was already efficient
        self.population = np.random.uniform(
            low=self.low_bound,
            high=self.high_bound,
            size=(self.pop_size, self.dna_size)
        )

    def select(self):
        """
        Fully vectorized standard uniform tournament selection.
        Note: The original implementation (weighted participant selection)
        is non-standard and slow. This is the efficient, common approach.
        """
        fitness = self.fitness_func(self.population)

        # 1. Select all tournament participants at once
        participant_indices = np.random.randint(0, self.pop_size,
                                                size=(self.pop_size, self.tournament_size))

        # 2. Get fitness for all participants
        participant_fitness = fitness[participant_indices]

        # 3. Find the index of the winner *within each tournament*
        winner_local_indices = np.argmax(participant_fitness, axis=1)

        # 4. Get the global population index of each winner
        winner_indices = participant_indices[np.arange(self.pop_size), winner_local_indices]

        # 5. Create the new population from winners
        self.population = self.population[winner_indices]

    def crossover(self):
        """
        Fully vectorized crossover with feature importance logic.
        """
        # 1. Get shuffled parent indices
        indices = np.arange(self.pop_size)
        np.random.shuffle(indices)

        parents1 = self.population[indices[::2]]
        parents2 = self.population[indices[1::2]]

        # 2. Create the base probability mask
        # Shape: (pop_size // 2, dna_size)
        base_mask = np.random.rand(self.pop_size // 2, self.dna_size) < self.crossover_rate

        # 3. Create the final mask based on fi_method
        if self.fi_method == "uniform":
            final_swap_mask = base_mask
        else:
            # Swap only if base_mask is True AND the feature is "important"
            # (dna_size,) mask broadcasts to (pop_size // 2, dna_size)
            final_swap_mask = base_mask & self.is_important_mask

        # 4. Create children using np.where
        child1 = np.where(final_swap_mask, parents2, parents1)
        child2 = np.where(final_swap_mask, parents1, parents2)

        self.population = np.vstack((child1, child2))

    def mutate(self):
        """
        Fully vectorized mutation with feature importance logic.
        """
        # 1. Create mask of genes that will be mutated
        # Shape: (pop_size, dna_size)
        trigger_mask = np.random.rand(self.pop_size, self.dna_size) < self.mutation_rate

        if self.fi_method == "uniform":
            # Standard GA: reset mutation for all triggered genes

            # Generate new random values for *all* genes
            reset_values = np.random.uniform(self.low_bound,
                                             self.high_bound,
                                             size=(self.pop_size, self.dna_size))

            # Only apply them where the trigger_mask is True
            self.population = np.where(trigger_mask, reset_values, self.population)

        else:
            # FIGA: Apply scaled mutation or reset mutation

            # --- Generate *all* potential new values ---

            # 1. Potential "scaled" mutation values (for important features)
            # (dna_size,) arrays broadcast to (pop_size, dna_size)
            scaled_deltas = np.random.uniform(-self.mutation_range,
                                              self.mutation_range,
                                              size=(self.pop_size, self.dna_size))
            potential_scaled_pop = self.population + scaled_deltas

            # 2. Potential "reset" mutation values (for non-important features)
            potential_reset_pop = np.random.uniform(self.low_bound,
                                                    self.high_bound,
                                                    size=(self.pop_size, self.dna_size))

            # --- Use masks to build the new population ---

            # First, choose between scaled/reset for *all* genes
            # This creates a full (pop_size, dna_size) array of "what-if" new values
            # (dna_size,) mask broadcasts to (pop_size, dna_size)
            new_values = np.where(self.is_important_mask,
                                  potential_scaled_pop,
                                  potential_reset_pop)

            # Now, use the trigger_mask to decide whether to use the
            # new_value or keep the original self.population value
            self.population = np.where(trigger_mask, new_values, self.population)

        # Finally, clip the entire population to ensure bounds are respected
        # (This is necessary for scaled mutations and harmless for resets)
        self.population = np.clip(self.population, self.low_bound, self.high_bound)

    def evolve(self):
        # This was already efficient
        self.select()
        self.crossover()
        self.mutate()