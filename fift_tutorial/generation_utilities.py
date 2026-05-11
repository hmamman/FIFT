"""
This python file provides essential functions for individual discrimination generation.
"""


import numpy as np
import itertools
import copy

from sklearn import cluster


def clip(instance, constraint):
    # clip the generated instance to satisfy the constraint

    return np.minimum(constraint[:, 1], np.maximum(constraint[:, 0], instance))

def clustering(data, c_num):
    # standard KMeans algorithm

    kmeans = cluster.KMeans(n_clusters=c_num)
    y_pred = kmeans.fit_predict(data)
    return [data[y_pred==n] for n in range(c_num)]


def random_pick(probability):
    # randomly pick an element from a probability distribution

    random_number = np.random.rand()
    current_proba = 0
    for i in range(len(probability)):
        current_proba += probability[i]
        if current_proba > random_number:
            return i


def get_seed(clustered_data, X_len, c_num, cluster_i, fashion='RoundRobin'):
    # get a seed from the specified cluster in a round-robin fashion
    # alternatively choose 'Distribution' to randomly sample a seed from a cluster with the probability proportional to the cluster size

    if fashion == 'RoundRobin':
        index = np.random.randint(0, len(clustered_data[cluster_i]))
        return clustered_data[cluster_i][index]
    elif fashion == 'Distribution':
        pick_probability = [len(clustered_data[i]) / X_len for i in range(c_num)]
        x = clustered_data[random_pick(pick_probability)]
        index = np.random.randint(0, len(x))
        return x[index], index

def similar_set(x, num_attribs, protected_attribs, constraint):
    similar_x = np.empty(shape=(0, num_attribs))
    x = np.array(x)
    protected_domain = []
    for i in protected_attribs:
        protected_domain.append(list(range(int(constraint[i][0]), int(constraint[i][1]) + 1)))
    all_combs = np.array(list(itertools.product(*protected_domain)))
    for comb in all_combs:
        x_new = copy.deepcopy(x)
        for a, c in zip(protected_attribs, comb):
            x_new[:, a] = c
        similar_x = np.append(similar_x, x_new, axis=0)
    return similar_x


def is_discriminatory(x, similar_x, model):
    # identify whether the instance is discriminatory w.r.t. the model
    y_pred = model.predict(x)
    y_similar = model.predict(similar_x)
    
    if np.any(y_similar != y_pred):
        return True
    return False