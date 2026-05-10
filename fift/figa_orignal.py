import copy
import itertools
import math
import os
import sys

import time

import joblib
import numpy as np

# Get the absolute path to the directory where figa.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# One levels up from figa.py
sys.path.append(os.path.join(base_path, "../"))

from figa_tutorial.ga_orignal import GA
from utils.helpers import get_experiment_params, generate_report, get_data
from utils.model_wrapper import model_wrapper
from figa_tutorial.compute_feature_importance import compute_feature_importance
from figa_tutorial import generation_utilities


class FIGA:
    def __init__(self, config, model, sensitive_param, classifier_name, population_size=200, crossover_rate=0.5, mutation_rate=0.1,
                 fi_method='shap', fi_threshold_type='25_percentile', duplicate_penalty=True,):
        self.start_time = time.time()
        self.config = config
        self.population_size = population_size

        self.tot_inputs = set()
        self.disc_inputs = set()

        self.constraint = np.array(self.config.input_bounds)
        self.sensitive_param = sensitive_param
        self.protected_attribs = np.array([1-sensitive_param])
        self.num_attribs = len(self.constraint)
        self.model = model_wrapper(model)
        self.classifier_name = classifier_name.lower()

        self.time_to_1000_disc = -1
        self.total_generated = 0
        self.cumulative_efficiency = []
        self.inference_count = 0

        self.log_interval = 300
        self.initial_log_interval = self.log_interval

        self.duplicate_penalty = duplicate_penalty


        self.setup = GA(
            pop_size=self.population_size,
            dna_size=len(self.config.input_bounds),
            bound=self.config.input_bounds,
            fitness_func=self.discrimination_object_func(),
            feature_importance=self.get_feature_importance(fi_method=fi_method),
            fi_method=fi_method,
            fi_threshold_type=fi_threshold_type,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate
        )

        self.approach_name = f'FIGA_uniform_crossover'
        # self.approach_name = f'FIGA_perm'

    def load_data(self):
        data = get_data(self.config.dataset_name)
        X, Y, _, _ = data()
        return X, Y

    def generate_seeds(self, data, c_num=4, num_seeds=1000, fashion='Distribution'):
        clustered_data = generation_utilities.clustering(data, c_num)
        indices = []
        seeds = []
        for i in range(100_000_000):
            x, index = generation_utilities.get_seed(clustered_data, len(data), c_num, i % c_num, fashion=fashion)
            seeds.append(x)
            indices.append(index)

            if len(seeds) >= num_seeds:
                break
        # print(indices)
        return np.array(seeds), indices

    def get_feature_importance(self, fi_method):
        data = get_data(self.config.dataset_name)
        X, Y, _, _ = data()
        feature_importance = compute_feature_importance(
            model=self.model, method=fi_method,
            X=X, y=Y
        )
        # Normalize the feature importance to sum to 1
        return feature_importance/np.sum(feature_importance)
    
    def clip(self, instance):
        # clip the generated instance to satisfy the constraint

        return np.minimum(self.constraint[:, 1], np.maximum(self.constraint[:, 0], instance))

    def similar_set(self, X):
        X = np.array(X)

        # Identify the protected attribute index
        protected_attribs = self.sensitive_param - 1

        # Get the range of values for the protected attribute
        low, high = self.constraint[protected_attribs]
        protected_domain = np.arange(int(low), int(high) + 1)

        # Create an array of the base instances, repeated for each value in the protected domain
        num_protected_values = len(protected_domain)
        num_X_instances = len(X)

        similar_X = np.repeat(X, num_protected_values, axis=0)

        # Create an array of the protected values, repeated for each instance in X
        protected_values_tiled = np.tile(protected_domain, num_X_instances)

        # Assign the new protected values in a single vectorized operation
        similar_X[:, protected_attribs] = protected_values_tiled

        return similar_X

    def discrimination_object_func(self):
        """
         Fairness objective with separate predictions:
         - one for the original inputs
         - one for the generated variants
         """

        def func(indvs):
            x_array = np.asarray(indvs, dtype=int)
            x_array = self.clip(x_array)
            N = len(x_array)

            if N == 0:
                return x_array, np.array([], dtype=int), np.array([], dtype=int)
            self.total_generated += N
            keys = [tuple(int(v) for v in row[:self.num_attribs]) for row in x_array]
            similar_x = self.similar_set(x_array)
            M = int(similar_x.shape[0] / N)  # number of variants per candidate

            y_base = self.model.predict(x_array).reshape(-1)
            y_variants = self.model.predict(similar_x).reshape(N, M)
            self.inference_count += 2

            # print(y_base[0], y_variants[0, :])

            # self.inference_count += 2
            unique_counts = np.fromiter(
                (np.unique(np.concatenate(([y_base[i]], y_variants[i,:]))).size for i in range(N)),
                dtype=int, count=N
            )

            # print(unique_counts)
            # Penalize duplicates (already in tot_inputs) ---
            if self.duplicate_penalty:
                for i, key in enumerate(keys):
                    if key in self.tot_inputs:
                        unique_counts[i] = 1  # redundant → force to 1
            self.tot_inputs.update(keys)
            # discriminatory candidates
            disc_idx = np.where(unique_counts > 1)[0]
            x_disc = x_array[disc_idx]
            if len(x_disc) > 0:
                self.disc_inputs.update(
                    tuple(int(v) for v in row[:self.num_attribs]) for row in x_disc
                )
                self.set_time_to_1000_disc()

            return unique_counts
        return func

    
    def evaluate_disc(self, inp):
        inp0 = np.array([int(x) for x in inp])
        inp1 = np.array([int(x) for x in inp])

        inp0 = inp0.reshape(1, -1)
        inp1 = inp1.reshape(1, -1)

        inp0 = np.asarray(inp0).reshape((1, -1))
        out0 = self.model.predict(inp0)

        for val in range(self.config.input_bounds[self.sensitive_param - 1][0],
                         self.config.input_bounds[self.sensitive_param - 1][1] + 1):
            if val != inp0[0][self.sensitive_param - 1]:
                inp1[0][self.sensitive_param - 1] = val
                out1 = self.model.predict(inp1)
               

                if out0 != out1:

                    return True
        return False
    
    def discrimination_check(self):
        def func(inp):
            x1 = np.array([int(x) for x in inp]).reshape(1, -1)
            x1 = generation_utilities.clip(x1, self.constraint)
        
            self.tot_inputs.add(tuple(map(tuple, x1)))

            self.total_generated += 1

            similar_x1 = generation_utilities.similar_set(x1, self.num_attribs, self.protected_attribs, self.constraint)
            if generation_utilities.is_discriminatory(x1, similar_x1, self.model):
                self.disc_inputs.add(tuple(map(tuple, x1)))
                self.set_time_to_1000_disc()
                return 1
            return 0
        return func
    
    def update_cumulative_efficiency(self, iteration):
        """
        Update the cumulative efficiency data if the current number of total inputs
        meets the tracking criteria (first input or every tracking_interval inputs).
        """
        total_inputs = len(self.tot_inputs)
        total_disc = len(self.disc_inputs)
        self.cumulative_efficiency.append([time.time() - self.start_time, iteration, total_inputs, total_disc])

    def set_time_to_1000_disc(self):
        disc_inputs_count = len(self.disc_inputs)

        if disc_inputs_count >= 1000 and self.time_to_1000_disc == -1:
            self.time_to_1000_disc = time.time() - self.start_time
            print(f"\nTime to generate 1000 discriminatory inputs: {self.time_to_1000_disc:.2f} seconds")

    def run(self, max_samples=1000 * 1000, max_allowed_time=3600):

        max_evolution = math.ceil(max_samples / self.population_size)

        for i in range(max_evolution):
            self.setup.evolve()
            self.update_cumulative_efficiency(i)

            use_time = time.time() - self.start_time
            if use_time >= self.log_interval:
                self.log_interval += self.initial_log_interval
                self.report(elapsed_time=use_time, is_log=True)

            if self.log_interval >= max_allowed_time:
                break

        elapsed_time = time.time() - self.start_time

        self.report(elapsed_time=elapsed_time, is_log=False)

    def report(self, elapsed_time, is_log: bool):
        additional_data = {'inference_count': self.inference_count}
        generate_report(
            approach_name=self.approach_name,
            dataset_name=self.config.dataset_name,
            classifier_name=self.classifier_name,
            sensitive_name=self.config.sens_name[self.sensitive_param],
            tot_inputs=self.tot_inputs,
            disc_inputs=self.disc_inputs,
            total_generated_inputs=self.total_generated,
            elapsed_time=elapsed_time,
            time_to_1000_disc=self.time_to_1000_disc,
            cumulative_efficiency=self.cumulative_efficiency,
            save_data=False,
            is_log=is_log,
            **additional_data,
        )


if __name__ == '__main__':
    config, sensitive_name, sensitive_param, classifier_name, max_allowed_time = get_experiment_params()

    print(f'Dataset: {config.dataset_name}')
    print(f'Classifier: {classifier_name}')
    print(f'Sensitive name: {sensitive_name}')
    print('')

    classifier_path = f'models/{config.dataset_name}/{classifier_name}.pkl'
    model = joblib.load(classifier_path)

    figa = FIGA(
        config=config,
        model=model,
        sensitive_param=sensitive_param,
        classifier_name=classifier_name,
    )

    figa.run(max_allowed_time=max_allowed_time)
