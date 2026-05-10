import math
import os
import sys
import numpy as np
import random
import time
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer

# Get the absolute path to the directory where expga.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Two levels up from expga.py
sys.path.append(os.path.join(base_path, "../../"))

from utils.helpers import get_data, generate_report, get_experiment_params
from baseline.expga.ga import GA

class ExpGA:
    def __init__(self, config, model, sensitive_param, threshold_l=10, threshold=0):
        self.approach_name = 'ExpGA'
        self.start_time = time.time()
        self.config = config
        self.global_disc_inputs = set()
        self.global_disc_inputs_list = []
        self.local_disc_inputs = set()
        self.local_disc_inputs_list = []
        self.tot_inputs = set()
        self.location = np.zeros(40)
        self.sensitive_param = sensitive_param
        self.threshold_l = threshold_l
        self.threshold = threshold
        self.input_bounds = self.config.input_bounds
        self.model = model

        self.time_to_1000_disc = -1
        self.total_generated = 0
        self.cumulative_efficiency = []
        self.tracking_interval = 1000

    def set_threshold_l(self):
        dataset_thresholds = {
            "census": 7,
            "bank": 10,
            "credit": 14,
            "meps": 10,
            "compas": 10
        }

        dataset_name = self.config.dataset_name
        if dataset_name in dataset_thresholds:
            self.threshold_l = dataset_thresholds[dataset_name]
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")

    def construct_explainer(self, train_vectors, feature_names, class_names):
        explainer = LimeTabularExplainer(train_vectors, feature_names=feature_names,
                                         class_names=class_names, discretize_continuous=False)
        return explainer

    def shap_value(self, test_vectors):
        background = shap.kmeans(test_vectors, 10)
        explainer = shap.KernelExplainer(self.model.predict_proba, background)
        shap_values = explainer.shap_values(test_vectors)
        return shap_values

    def search_seed(self, feature_names, sens_name, explainer, train_vectors, num, X_ori):
        seed = []
        for x in train_vectors:
            self.tot_inputs.add(tuple(x))

            self.total_generated += 1

            proba_func = self.model.predict_proba

            exp = explainer.explain_instance(x, proba_func, num_features=num)
            explain_labels = exp.available_labels()
            exp_result = exp.as_list(label=explain_labels[0])
            rank = [item[0] for item in exp_result]
            loc = rank.index(sens_name)
            self.location[loc] += 1
            if loc < self.threshold_l:
                seed.append(x)
            if len(seed) >= 200:
                return seed
        return seed

    def search_seed_shap(self, feature_names, sens_name, shap_values, train_vectors):
        seed = []
        for i in range(len(shap_values[0])):
            sample = shap_values[0][i]
            sorted_shap_value = sorted(
                [[feature_names[j], sample[j]] for j in range(len(sample))],
                key=lambda x: abs(x[1]), reverse=True
            )
            rank = [item[0] for item in sorted_shap_value]
            loc = rank.index(sens_name)
            if loc < 10:
                seed.append(train_vectors[i])
            if len(seed) > 10:
                return seed
        return seed

    class GlobalDiscovery:
        def __init__(self, stepsize=1):
            self.stepsize = stepsize

        def __call__(self, iteration, params, input_bounds, sensitive_param):
            samples = []
            while len(samples) < iteration:
                x = np.zeros(params)
                for i in range(params):
                    random.seed(time.time())
                    x[i] = random.randint(input_bounds[i][0], input_bounds[i][1])
                x[sensitive_param - 1] = 0
                samples.append(x)
            return samples

    def evaluate_global(self, inp):
        inp0 = [int(i) for i in inp]
        inp1 = [int(i) for i in inp]

        value = random.randint(self.config.input_bounds[self.sensitive_param - 1][0],
                               self.config.input_bounds[self.sensitive_param - 1][1])
        inp1[self.sensitive_param - 1] = value

        inp0 = np.asarray(inp0).reshape((1, -1))
        inp1 = np.asarray(inp1).reshape((1, -1))

        out0 = self.make_prediction(inp0)
        out1 = self.make_prediction(inp1)

        self.tot_inputs.add(tuple(map(tuple, inp0)))
        self.total_generated += 1

        if abs(out0 - out1) > self.threshold and tuple(map(tuple, inp0)) not in self.global_disc_inputs:
            self.global_disc_inputs.add(tuple(map(tuple, inp0)))
            self.global_disc_inputs_list.append(inp0.tolist()[0])

            self.set_time_to_1000_disc()

        return abs(out1 + out0)

    def evaluate_local(self, inp):
        inp0 = [int(i) for i in inp]

        self.tot_inputs.add(tuple(inp0))
        self.total_generated += 1

        out0 = 0
        out1 = 0

        for val in range(self.config.input_bounds[self.sensitive_param - 1][0],
                         self.config.input_bounds[self.sensitive_param - 1][1] + 1):
            if val != inp[self.sensitive_param - 1]:
                inp1 = [int(i) for i in inp]
                inp1[self.sensitive_param - 1] = val

                inp0 = np.asarray(inp0).reshape((1, -1))
                inp1 = np.asarray(inp1).reshape((1, -1))

                out0 = self.make_prediction(inp0)
                out1 = self.make_prediction(inp1)

                if abs(out0 - out1) > self.threshold and (tuple(map(tuple, inp0)) not in self.global_disc_inputs) \
                        and (tuple(map(tuple, inp0)) not in self.local_disc_inputs):
                    self.local_disc_inputs.add(tuple(map(tuple, list(inp0))))
                    self.local_disc_inputs_list.append(inp0.tolist()[0])

                    self.set_time_to_1000_disc()

                    return 2 * abs(out1 - out0) + 1
        return 2 * abs(out1 - out0) + 1

    def make_prediction(self, inp):
        output = self.model.predict(inp)

        return (output > 0.5).astype(int)

    def update_cumulative_efficiency(self, iteration):
        """
                Update the cumulative efficiency data if the current number of total inputs
                meets the tracking criteria (first input or every tracking_interval inputs).
                """
        total_inputs = len(self.tot_inputs)
        total_disc = len(self.local_disc_inputs) + len(self.global_disc_inputs)
        self.cumulative_efficiency.append([time.time() - self.start_time, iteration, total_inputs, total_disc])

    def set_time_to_1000_disc(self):
        disc_inputs_count = len(self.global_disc_inputs) + len(self.local_disc_inputs)
        if disc_inputs_count >= 1000 and self.time_to_1000_disc == -1:
            self.time_to_1000_disc = time.time() - self.start_time
            print(f"\nTime to generate 1000 discriminatory inputs: {self.time_to_1000_disc:.2f} seconds")

    def run(self, max_global=1000, max_local=1000, max_allowed_time=3600):
        self.start_time = time.time()
        self.total_generated = 0
        self.set_threshold_l()

        feature_names = self.config.feature_name
        class_names = self.config.class_name
        sens_name = self.config.sens_name[self.sensitive_param]
        params = self.config.params

        data = get_data(self.config.dataset_name)
        X, Y, input_shape, nb_classes = data()

        global_discovery = self.GlobalDiscovery()

        train_samples = global_discovery(max_global, params, self.input_bounds, self.sensitive_param)
        train_samples = np.array(train_samples)
        np.random.shuffle(train_samples)

        explainer = self.construct_explainer(X, feature_names, class_names)
        seed = self.search_seed(feature_names, sens_name, explainer, train_samples, params, X)
        print('Finish Searchseed')

        for inp in seed:
            inp0 = np.asarray([int(i) for i in inp]).reshape((1, -1))
            self.global_disc_inputs.add(tuple(map(tuple, inp0)))
            self.global_disc_inputs_list.append(inp0.tolist()[0])
            self.set_time_to_1000_disc()

        print("Finished Global Search")
        print('length of total input is:' + str(len(self.tot_inputs)))
        print('length of global discovery is:' + str(len(self.global_disc_inputs_list)))

        end = time.time()

        print('Total time:' + str(end - self.start_time))

        print("")
        print("Starting Local Search")

        nums = self.global_disc_inputs_list

        DNA_SIZE = len(self.input_bounds)
        ga = GA(
            nums=nums,
            bound=self.input_bounds,
            func=self.evaluate_local,
            DNA_SIZE=DNA_SIZE,
        )

        ga.population = np.array(seed)

        count = 300

        max_iter = math.ceil(((max_local * max_global) - self.total_generated) / len(nums))

        for i in range(max_iter):
            ga.evolution()
            self.update_cumulative_efficiency(i)
            end = time.time()
            use_time = end - self.start_time
            if use_time >= count:
                count += 300
                self.report(elapsed_time=use_time, is_log=True)

            if count >= max_allowed_time:
                break

        elapsed_time = time.time() - self.start_time

        self.report(elapsed_time=elapsed_time, is_log=False)

    def report(self, elapsed_time, is_log: bool):
        disc_inputs = self.local_disc_inputs | self.global_disc_inputs

        generate_report(
            approach_name=f'{self.approach_name}',
            dataset_name=self.config.dataset_name,
            classifier_name=self.model.__class__.__name__,
            sensitive_name=self.config.sens_name[self.sensitive_param],
            tot_inputs=self.tot_inputs,
            disc_inputs=disc_inputs,
            total_generated_inputs=self.total_generated,
            elapsed_time=elapsed_time,
            time_to_1000_disc=self.time_to_1000_disc,
            cumulative_efficiency=self.cumulative_efficiency,
            is_log=is_log,
        )


if __name__ == '__main__':
    config, sensitive_name, sensitive_param, classifier_name, max_allowed_time = get_experiment_params()

    print(f'Dataset: {config.dataset_name}')
    print(f'Classifier: {classifier_name}')
    print(f'Sensitive name: {sensitive_name}')
    print('')

    classifier_path = f'models/{config.dataset_name}/{classifier_name}.pkl'
    model = joblib.load(classifier_path)

    expga = ExpGA(
        config=config,
        model=model,
        sensitive_param=sensitive_param
    )

    expga.run(max_allowed_time=max_allowed_time)
