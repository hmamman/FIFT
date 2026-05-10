# (BEGIN maft.py)
import time
import numpy as np
import tensorflow as tf
import sys, os

# Adjust the import path as needed
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, "../../"))
from utils.helpers import get_data, generate_report, get_experiment_params
from baseline.maft import generation_utilities

# allocate GPU and set dynamic memory growth
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class MAFT:
    def __init__(self, config, model, sensitive_param, cluster_num=4, max_global=1000, max_local=1000, max_iter=10):
        self.approach_name = 'MAFT'
        self.start_time = time.time()

        self.config = config
        self.constraint = np.array(self.config.input_bounds)
        self.sensitive_param = sensitive_param
        self.protected_attribs = np.array([sensitive_param-1])
        self.num_attribs = len(self.constraint)
        self.model = model
        self.cluster_num = cluster_num
        self.max_global = max_global
        self.max_local = max_local
        self.max_iter = max_iter
        self.perturbation_size = 1

        # Initialize testing results storage
        self.tot_inputs = set()
        self.disc_inputs = set()
        self.disc_inputs_list = []

        self.dataset_name = self.config.dataset_name

        self.time_to_1000_disc = -1
        self.total_generated = 0
        self.cumulative_efficiency = []
        self.inference_count = 0
        self.gradient_calls = 0
        self.log_interval = 300
        self.initial_log_interval = self.log_interval

        # Prediction cache to avoid redundant model calls (quantized keys)
        self.pred_cache = {}
        self.cache_round = 8  # decimals to round continuous features for caching

        # seen inputs hash to prevent duplicates (quantized)
        self.seen_hash = set()



    def compute_grad(self, x, perturbation_size=1e-4):
        # Backwards-compatible wrapper for single seed
        grads = self.compute_grad_batch(np.atleast_2d(x), perturbation_size=perturbation_size)
        return grads[0]


    def compute_grad_batch(self, X_seeds, perturbation_size=1e-4, chunk_size=4096):
        """
        Vectorized finite-difference gradient approximation for multiple seeds.
        X_seeds: np.array shape (N, d)
        Returns: gradients np.array shape (N, d)
        """
        X_seeds = np.asarray(X_seeds, dtype=float)
        N, d = X_seeds.shape
        h = perturbation_size
        # Build offsets: for each seed we need (1 + d) rows: [x, x+e1, ..., x+ed]
        # We'll construct base repeated and offsets tiled.
        offsets = np.zeros((d + 1, d), dtype=float)
        offsets[1:] = np.eye(d) * h
        base = np.repeat(X_seeds, repeats=(d + 1), axis=0)  # (N*(d+1), d)
        full_offsets = np.tile(offsets, (N, 1))  # (N*(d+1), d)
        batch = base + full_offsets
        # Predict in chunks to avoid memory spikes
        preds = []
        for i in range(0, batch.shape[0], chunk_size):
            sub = batch[i:i+chunk_size]
            sub_tf = tf.constant(sub, dtype=tf.float32)
            p = self.model(sub_tf).numpy().reshape(-1)
            preds.append(p)
            self.inference_count += 1
        preds = np.concatenate(preds, axis=0)
        # reshape to (N, d+1)
        preds = preds.reshape(N, d + 1)
        y0 = preds[:, 0:1]  # (N,1)
        grads = (preds[:, 1:] - y0) / h  # (N, d)
        # signed by baseline class: if base prob <=0.5 we keep grads, else negate
        signed_grads = np.where(y0 <= 0.5, grads, -grads)
        self.gradient_calls += 1
        return signed_grads


    def _quantize_key(self, x):
        return tuple(np.round(x, self.cache_round).tolist())

    def batch_predict_with_cache(self, X_batch, chunk_size=4096):
        """
        Predict probabilities for X_batch (np.array shape (M,d)), using cache to avoid duplicate model calls.
        Returns: np.array shape (M,) of probabilities.
        """
        X_batch = np.asarray(X_batch, dtype=float)
        M = X_batch.shape[0]
        out = np.empty((M,), dtype=float)
        to_query_idx = []
        to_query_X = []
        for i, x in enumerate(X_batch):
            key = self._quantize_key(x)
            if key in self.pred_cache:
                out[i] = self.pred_cache[key]
            else:
                to_query_idx.append(i)
                to_query_X.append(x)
        if to_query_X:
            batch = np.vstack(to_query_X)
            preds = []
            for i in range(0, batch.shape[0], chunk_size):
                sub = batch[i:i+chunk_size]
                sub_tf = tf.constant(sub, dtype=tf.float32)
                p = self.model(sub_tf).numpy().reshape(-1)
                preds.append(p)
                self.inference_count += 1
            preds = np.concatenate(preds, axis=0)
            for idx, p, x in zip(to_query_idx, preds, to_query_X):
                key = self._quantize_key(x)
                self.pred_cache[key] = float(p)
                out[idx] = float(p)
        return out

    def is_discriminatory(self, x, similar_x):
        # Batched discriminatory check with cache. similar_x is array-like of candidates.
        batch = np.vstack([x] + list(similar_x))
        probs = self.batch_predict_with_cache(batch)
        base_label = probs[0] > 0.5
        # early exit on first mismatch
        for p in probs[1:]:
            if (p > 0.5) != base_label:
                return True
        return False

    def max_diff(self, x, similar_x):
        # Choose the similar instance whose model probability differs most from x (L2 on probs).
        batch = np.vstack([x] + list(similar_x))
        probs = self.batch_predict_with_cache(batch)
        y_base = probs[0]
        diffs = (y_base - probs[1:]) ** 2
        idx = int(np.argmax(diffs))
        return np.asarray(similar_x)[idx]


    def maft_generation(self,X, seeds, decay=0.5, update_interval=5, s_g=1.0, s_l=1.0, epsilon=1e-6, max_allowed_time=3600):
        # perform global generation and local generation successively on each single seed
        direction = [-1, 1]
        for index, instance in enumerate(seeds):
            use_time = time.time() - self.start_time
            if use_time >= self.log_interval:
                self.log_interval += self.initial_log_interval
                self.report(self.tot_inputs, self.disc_inputs, self.total_generated, self.inference_count)

            x1 = instance.copy().astype(float)
            self.tot_inputs.add(tuple(x1))
            self.total_generated += 1
            # global iteration
            for g in range(self.max_global):
                # 1. randomly pick a partner in the population to form pair
                x_pair = generation_utilities.random_pair(x1, self.constraint)
                # 2. check whether they are discriminatory
                if self.is_discriminatory(x1, [x_pair]):
                    self.disc_inputs.add(tuple(x1))
                    self.disc_inputs_list.append(tuple(x1))
                    self.set_time_to_1000_disc()
                    break
                # 3. local search start
                # choose a partner using the max_diff function
                x2 = self.max_diff(x1, [x_pair])
                # prepare gradients
                grad1 = self.compute_grad(x1)
                grad2 = self.compute_grad(x2)
                # direction of gradient agreement except protected features
                direction_g = np.zeros_like(x1)
                sign_grad1 = np.sign(grad1)
                sign_grad2 = np.sign(grad2)
                for attrib in range(self.num_attribs):
                    if attrib not in self.protected_attribs and sign_grad1[attrib] == sign_grad2[attrib]:
                        direction_g[attrib] = (-1) * sign_grad1[attrib]
                x1 = x1 + s_g * direction_g
                x1 = generation_utilities.clip(x1, self.constraint)

                self.tot_inputs.add(tuple(x1))
                self.total_generated += 1

                # local search
                for _ in range(self.max_local):
                    # compute similar set (neighbours) - this block is from original implementation
                    similar_x1 = generation_utilities.similar_set(x1, self.num_attribs, self.protected_attribs, self.constraint)
                    if len(similar_x1) == 0:
                        break
                    x2 = self.max_diff(x1, similar_x1)
                    grad1 = decay * grad1 + self.compute_grad(x1)
                    grad2 = decay * grad2 + self.compute_grad(x2)
                    direction_g = np.zeros_like(x1)
                    sign_grad1 = np.sign(grad1)
                    sign_grad2 = np.sign(grad2)
                    for attrib in range(self.num_attribs):
                        if attrib not in self.protected_attribs and sign_grad1[attrib] == sign_grad2[attrib]:
                            direction_g[attrib] = (-1) * sign_grad1[attrib]
                    x1 = x1 + s_g * direction_g
                    x1 = generation_utilities.clip(x1, self.constraint)

                    self.tot_inputs.add(tuple(x1))
                    self.total_generated += 1

                    if self.is_discriminatory(x1, similar_x1):
                        self.disc_inputs.add(tuple(x1))
                        self.disc_inputs_list.append(tuple(x1))
                        self.set_time_to_1000_disc()
                        break

                # if len(self.disc_inputs) > 0:
                #     break

            # log progress
            if index % 10 == 0:
                print(f"[{self.approach_name}] generated: {len(self.tot_inputs)}  discriminatory: {len(self.disc_inputs)} total_generated: {self.total_generated} inference_count: {self.inference_count}")

            self.update_cumulative_efficiency(index)

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

    def run(self, max_allowed_time=3600):
        
        data = get_data(self.config.dataset_name)
        X, Y, input_shape, nb_classes = data()

        clustered_data = generation_utilities.clustering(X, self.cluster_num)
        seeds = np.empty(shape=(0, len(X[0])))
        for i in range(self.max_global):
            new_seed = generation_utilities.get_seed(clustered_data, len(X), self.cluster_num, i%self.cluster_num)
            seeds = np.append(seeds, [new_seed], axis=0)

        self.maft_generation(seeds=seeds, X=X, max_allowed_time=max_allowed_time)
        # self.report(elapsed_time=time.time()-self.start_time, is_log=False)

    def report(self, elapsed_time, is_log: bool):
        additional_data = {
            'inference_count': self.inference_count,
            'gradient_calls': self.gradient_calls
        }
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
            **additional_data,
        )


if __name__ == '__main__':
    config, sensitive_name, sensitive_param, classifier_name, max_allowed_time = get_experiment_params()

    print(f'Dataset: {config.dataset_name}')
    print(f'Classifier: {classifier_name}')
    print(f'Sensitive name: {sensitive_name}')
    print('')

    classifier_path = f'models/{config.dataset_name}/{classifier_name}.keras'
    model = tf.keras.models.load_model(classifier_path)

    maft = MAFT(
        config=config,
        model=model,
        sensitive_param=sensitive_param,
    )

    maft.run(max_allowed_time=max_allowed_time)
