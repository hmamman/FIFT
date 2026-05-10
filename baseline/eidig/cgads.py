import time
import numpy as np
import tensorflow as tf
import sys, os
import copy
from scipy.optimize import basinhopping
from sklearn.cluster import KMeans

# Adjust the import path as needed
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, "../../"))
from utils.helpers import get_data, generate_report, get_experiment_params

# allocate GPU and set dynamic memory growth
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# make outputs stable across runs for validation
np.random.seed(42)
tf.random.set_seed(42)


class NeuronFair:
    def __init__(self, config, model, sensitive_param, cluster_num=4, max_global=1000, max_local=1000, max_iter=40,
                 alpha=0.5):
        self.approach_name = 'NeuronFair'
        self.start_time = time.time()

        self.config = config
        self.constraint = np.array(self.config.input_bounds)
        self.sensitive_param = sensitive_param
        self.num_attribs = len(self.constraint)
        self.model = model
        self.cluster_num = cluster_num
        self.max_global = max_global
        self.max_local = max_local
        self.max_iter = max_iter
        self.alpha = alpha
        self.perturbation_size = 1

        # Initialize testing results storage
        self.tot_inputs = set()
        self.disc_inputs = set()
        self.global_disc_inputs = set()
        self.local_disc_inputs = set()
        self.disc_inputs_list = []

        self.dataset_name = self.config.dataset_name

        self.time_to_1000_disc = -1
        self.total_generated = 0
        self.cumulative_efficiency = []
        self.log_interval = 30
        self.initial_log_interval = self.log_interval

        # Get neuron weights and gradients
        self._setup_neuron_analysis()

    def _setup_neuron_analysis(self):
        """Setup neuron analysis for gradient computation"""
        # Find the ReLU layer for neuron analysis
        self.relu_layer_name = None
        for i, layer in enumerate(self.model.layers):
            if 'relu' in layer.name.lower() or isinstance(layer, tf.keras.layers.ReLU):
                self.relu_layer_name = layer.name
                break

        if self.relu_layer_name is None:
            # If no ReLU layer found, use the last hidden layer
            for layer in reversed(self.model.layers[:-1]):
                if hasattr(layer, 'activation'):
                    self.relu_layer_name = layer.name
                    break

    def get_neuron_weights(self, X):
        """
        Get neuron weights based on sensitive attribute variations
        """
        # Create model that outputs intermediate layer
        intermediate_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=self.model.get_layer(self.relu_layer_name).output
        )

        # Create alternative samples with flipped sensitive attribute
        nX = copy.copy(X)
        sens_bounds = self.constraint[self.sensitive_param - 1]

        # Flip sensitive attribute values
        eq_mask = (nX[:, self.sensitive_param - 1] == sens_bounds[0])
        nX[eq_mask, self.sensitive_param - 1] = sens_bounds[1]
        nX[~eq_mask, self.sensitive_param - 1] = sens_bounds[0]

        # Get activations
        sa = intermediate_model(X)
        nsa = intermediate_model(nX)

        # Compute sensitivity features
        sf = tf.reduce_mean(tf.abs(sa) + tf.abs(nsa), axis=0)

        # Select top alpha proportion of neurons
        num_neurons = int(self.alpha * tf.shape(sf)[0])
        threshold_idx = tf.nn.top_k(sf, k=num_neurons).indices[-1] if num_neurons > 0 else 0
        threshold = tf.gather(sf, threshold_idx)

        weights = tf.cast(sf >= threshold, tf.float32)
        return weights.numpy()

    def compute_neuron_gradients(self, x, nx, weights):
        """
        Compute gradients with respect to neuron weights
        """
        x_tensor = tf.constant([x], dtype=tf.float32)
        nx_tensor = tf.constant([nx], dtype=tf.float32)

        # Get intermediate layer model
        intermediate_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer(self.relu_layer_name).output
        )

        with tf.GradientTape() as tape1:
            tape1.watch(x_tensor)
            x_hidden = intermediate_model(x_tensor)
            x_weighted = tf.reduce_sum(x_hidden * weights)

        with tf.GradientTape() as tape2:
            tape2.watch(nx_tensor)
            nx_hidden = intermediate_model(nx_tensor)
            nx_weighted = tf.reduce_sum(nx_hidden * weights)

        x_grad = tape1.gradient(x_weighted, x_tensor)
        nx_grad = tape2.gradient(nx_weighted, nx_tensor)

        return x_grad[0].numpy(), nx_grad[0].numpy()

    def check_discrimination(self, x):
        """
        Check whether the test case is discriminatory
        """
        x = x.astype(int)
        x_tensor = tf.constant([x], dtype=tf.float32)

        label = tf.argmax(self.model(x_tensor), axis=1)[0].numpy()

        # Check all possible values of sensitive feature
        sens_bounds = self.constraint[self.sensitive_param - 1]
        for val in range(sens_bounds[0], sens_bounds[1] + 1):
            if val != x[self.sensitive_param - 1]:
                x_new = copy.deepcopy(x)
                x_new[self.sensitive_param - 1] = val
                x_new_tensor = tf.constant([x_new], dtype=tf.float32)
                label_new = tf.argmax(self.model(x_new_tensor), axis=1)[0].numpy()
                if label_new != label:
                    return True
        return False

    def clip(self, x):
        """Clip input to valid bounds"""
        for i in range(len(x)):
            x[i] = max(x[i], self.constraint[i][0])
            x[i] = min(x[i], self.constraint[i][1])
        return x

    def seed_test_input(self, clusters, limit):
        """Select seed inputs from clusters"""
        rows = []
        max_size = max([len(cluster) for cluster in clusters])
        i = 0

        while i < max_size and len(rows) < limit:
            for cluster in clusters:
                if i >= len(cluster) or len(rows) >= limit:
                    continue
                rows.append(cluster[i])
            i += 1

        return np.array(rows)

    def clustering(self, X):
        """Perform K-means clustering on the dataset"""
        kmeans = KMeans(n_clusters=self.cluster_num, random_state=42)
        labels = kmeans.fit_predict(X)

        clusters = []
        for i in range(self.cluster_num):
            cluster_indices = np.where(labels == i)[0]
            clusters.append(cluster_indices)

        return clusters

    class LocalPerturbation:
        """Local perturbation for basin hopping"""

        def __init__(self, parent, n_value, weights):
            self.parent = parent
            self.n_value = n_value
            self.weights = weights
            self.perturbation_size = 1

        def softmax(self, m):
            probs = np.exp(m - np.max(m))
            probs /= np.sum(probs)
            return probs

        def __call__(self, x):
            s = np.random.choice([1.0, -1.0]) * self.perturbation_size

            # Create paired sample with different sensitive attribute
            n_x = x.copy()
            n_x[self.parent.sensitive_param - 1] = self.n_value

            # Compute gradients
            try:
                x_grad, nx_grad = self.parent.compute_neuron_gradients(x, n_x, self.weights)
            except:
                # Fallback to uniform probability
                probs = np.ones(self.parent.num_attribs) / (self.parent.num_attribs - 1)
                probs[self.parent.sensitive_param - 1] = 0
                probs = probs / probs.sum()
                index = np.random.choice(range(self.parent.num_attribs), p=probs)
                local_cal_grad = np.zeros(self.parent.num_attribs)
                local_cal_grad[index] = 1.0
                return self.parent.clip(x + s * local_cal_grad).astype(int)

            # Check for zero gradients
            if np.allclose(x_grad, 0) and np.allclose(nx_grad, 0):
                probs = np.ones(self.parent.num_attribs) / (self.parent.num_attribs - 1)
                probs[self.parent.sensitive_param - 1] = 0
            else:
                # Normalize reciprocal of gradients
                grad_sum = 1.0 / (np.abs(x_grad) + np.abs(nx_grad) + 1e-8)
                grad_sum[self.parent.sensitive_param - 1] = 0
                probs = grad_sum / np.sum(grad_sum)

            probs = probs / probs.sum()

            try:
                index = np.random.choice(range(self.parent.num_attribs), p=probs)
            except:
                index = np.random.choice(range(self.parent.num_attribs))

            local_cal_grad = np.zeros(self.parent.num_attribs)
            local_cal_grad[index] = 1.0

            return self.parent.clip(x + s * local_cal_grad).astype(int)

    def evaluate_local(self, x, local_perturbation):
        """Evaluate local perturbation result"""
        result = self.check_discrimination(x)

        temp = x.astype(int).tolist()
        temp_key = temp[:self.sensitive_param - 1] + temp[self.sensitive_param:]

        self.tot_inputs.add(tuple(temp_key))
        self.total_generated += 1

        if result and tuple(temp_key) not in self.disc_inputs:
            self.disc_inputs.add(tuple(temp_key))
            self.local_disc_inputs.add(tuple(temp_key))
            self.set_time_to_1000_disc()

        return 1 if result else 0

    def neuronfair_generation(self, X, seeds, weights, max_allowed_time=3600):
        """Main NeuronFair generation algorithm"""

        for num, seed_idx in enumerate(seeds):
            use_time = time.time() - self.start_time
            if use_time >= self.log_interval:
                self.log_interval += self.initial_log_interval
                self.report(elapsed_time=use_time, is_log=True)

            if use_time >= max_allowed_time or self.total_generated >= self.max_global * self.max_local:
                break

            sample = X[seed_idx:seed_idx + 1].copy()

            # Global perturbation
            for iter_count in range(self.max_iter + 1):
                use_time = time.time() - self.start_time
                if use_time >= self.log_interval:
                    self.log_interval += self.initial_log_interval
                    self.report(elapsed_time=use_time, is_log=True)

                if use_time >= max_allowed_time or self.total_generated >= self.max_global * self.max_local:
                    break

                # Get model predictions
                sample_tensor = tf.constant(sample, dtype=tf.float32)
                probs = self.model(sample_tensor)[0].numpy()
                label = np.argmax(probs)
                prob = probs[label]

                # Find instance with maximum probability difference
                max_diff = 0
                n_value = -1
                sens_bounds = self.constraint[self.sensitive_param - 1]

                for i in range(sens_bounds[0], sens_bounds[1] + 1):
                    if i != sample[0][self.sensitive_param - 1]:
                        n_sample = sample.copy()
                        n_sample[0][self.sensitive_param - 1] = i
                        n_sample_tensor = tf.constant(n_sample, dtype=tf.float32)
                        n_probs = self.model(n_sample_tensor)[0].numpy()
                        n_label = np.argmax(n_probs)
                        n_prob = n_probs[n_label]

                        if label != n_label:
                            n_value = i
                            break
                        else:
                            prob_diff = abs(prob - n_prob)
                            if prob_diff > max_diff:
                                max_diff = prob_diff
                                n_value = i

                # Check if discriminatory
                temp = sample[0].astype(int).tolist()
                temp_key = temp[:self.sensitive_param - 1] + temp[self.sensitive_param:]

                if label != n_label and tuple(temp_key) not in self.disc_inputs:
                    self.disc_inputs.add(tuple(temp_key))
                    self.global_disc_inputs.add(tuple(temp_key))
                    self.set_time_to_1000_disc()

                    # Start local perturbation
                    local_perturbation = self.LocalPerturbation(self, n_value, weights)

                    def objective(x):
                        return self.evaluate_local(x, local_perturbation)

                    minimizer = {"method": "L-BFGS-B"}
                    basinhopping(objective, sample[0], stepsize=1.0,
                                 take_step=local_perturbation,
                                 minimizer_kwargs=minimizer,
                                 niter=self.max_local)
                    break

                # Global perturbation step
                n_sample = sample.copy()
                n_sample[0][self.sensitive_param - 1] = n_value

                try:
                    s_grad, n_grad = self.compute_neuron_gradients(sample[0], n_sample[0], weights)
                    s_grad_sign = np.sign(s_grad)
                    n_grad_sign = np.sign(n_grad)

                    # Find features with same impact
                    if np.allclose(s_grad_sign, 0):
                        g_diff = n_grad_sign
                    elif np.allclose(n_grad_sign, 0):
                        g_diff = s_grad_sign
                    else:
                        g_diff = (s_grad_sign == n_grad_sign).astype(float)

                    g_diff[self.sensitive_param - 1] = 0

                    if np.allclose(g_diff, 0):
                        # Random perturbation
                        delta = self.perturbation_size
                        s_grad_sign = np.random.randint(-delta, delta + 1, s_grad_sign.shape)
                        g_diff = np.ones(self.num_attribs)
                        g_diff[self.sensitive_param - 1] = 0

                    cal_grad = s_grad_sign * g_diff
                    sample[0] = self.clip(sample[0] + self.perturbation_size * cal_grad).astype(int)
                    self.total_generated += 1

                except Exception as e:
                    # Fallback random perturbation
                    random_grad = np.random.randint(-1, 2, self.num_attribs)
                    random_grad[self.sensitive_param - 1] = 0
                    sample[0] = self.clip(sample[0] + self.perturbation_size * random_grad).astype(int)
                    self.total_generated += 1

            self.update_cumulative_efficiency(num)

    def update_cumulative_efficiency(self, iteration):
        """Update cumulative efficiency tracking"""
        total_inputs = len(self.tot_inputs)
        total_disc = len(self.disc_inputs)
        self.cumulative_efficiency.append([time.time() - self.start_time, iteration, total_inputs, total_disc])

    def set_time_to_1000_disc(self):
        """Track time to reach 1000 discriminatory inputs"""
        disc_inputs_count = len(self.disc_inputs)
        if disc_inputs_count >= 1000 and self.time_to_1000_disc == -1:
            self.time_to_1000_disc = time.time() - self.start_time
            print(f"\nTime to generate 1000 discriminatory inputs: {self.time_to_1000_disc:.2f} seconds")

    def run(self, max_allowed_time=3600):
        """Main execution function"""
        data = get_data(self.config.dataset_name)
        X, Y, input_shape, nb_classes = data()

        # Get neuron weights
        weights = self.get_neuron_weights(X)

        # Perform clustering
        clusters = self.clustering(X)

        # Select seed inputs
        seeds = self.seed_test_input(clusters, min(self.max_global, len(X)))

        # Run NeuronFair generation
        self.neuronfair_generation(X, seeds, weights, max_allowed_time)

        # Final report
        self.report(elapsed_time=time.time() - self.start_time, is_log=False)

    def report(self, elapsed_time, is_log: bool):
        """Generate and print report"""
        generate_report(
            approach_name=self.approach_name,
            dataset_name=self.config.dataset_name,
            classifier_name=self.model.__class__.__name__,
            sensitive_name=self.config.sens_name[self.sensitive_param],
            tot_inputs=self.tot_inputs,
            disc_inputs=self.disc_inputs,
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

    classifier_path = f'models/{config.dataset_name}/{classifier_name}_slfc.keras'
    model = tf.keras.models.load_model(classifier_path)

    neuronfair = NeuronFair(
        config=config,
        model=model,
        sensitive_param=sensitive_param,
    )

    neuronfair.run(max_allowed_time=max_allowed_time)