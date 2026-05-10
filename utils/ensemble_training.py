"""
This python file trains the ensemble models for relabeling.
Five classifiers are trained for majority voting, including KNN, MLP, RBF SVM, Random Forest, Naive Bayes.
"""

import sys
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone


# Get the absolute path to the directory where main.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Two levels up from expga.py
sys.path.append(os.path.join(base_path, "../"))

from utils import helpers


# create classifiers
knn_clf = KNeighborsClassifier()
mlp_clf = MLPClassifier()
svm_clf = SVC(probability=True)
rf_clf = RandomForestClassifier()
nb_clf = GaussianNB()


# ensemble above classifiers for majority voting
eclf = VotingClassifier(estimators=[('knn', knn_clf), ('mlp', mlp_clf), ('svm', svm_clf), ('rf', rf_clf), ('nb', nb_clf)],
                        voting='soft')


# set a pipeline to handle the prediction process
ensemble_pipeline = Pipeline([('scaler', StandardScaler()),
                ('ensemble', eclf)])


def train_and_evaluate(data, config):
    dataset_name = config.dataset_name
    X, y, input_shape, nb_classes = data()

    # Identify protected attributes (indices are 1-based in config)
    protected_attribs = [idx - 1 for idx in config.sens_name.keys()]
    
    # Remove protected features from the dataset
    X = np.delete(X, protected_attribs, axis=1)
    print(f"[{dataset_name}] Removed {len(protected_attribs)} protected features. New shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use a fresh clone to ensure no state leakage between datasets
    model = clone(ensemble_pipeline)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"VEC Model's accuracy on {dataset_name} dataset is: {accuracy:.4f}")

    # Save the model to disk
    dir_path = f"models/ensemble"
    os.makedirs(dir_path, exist_ok=True)
    model_path = f'{dir_path}/{dataset_name}_ensemble.pkl'
    joblib.dump(model, model_path)
    print(f"Ensemble Model saved to {model_path}")
    # Bank          0.8988
    # Census        0.8431
    # Credit        0.7583
    # Compas        0.8209
    # Meps          0.8683   


if __name__ == "__main__":
    configs = helpers.get_config_dict()
    datasets = configs.keys()

    for dataset_name in datasets:
        # Check for dataset file in the root data folder
        data_path = os.path.join(base_path, "../data", f"{dataset_name}.py")
        if not os.path.exists(data_path):
            print(f"Dataset not found: {data_path}")
            continue # Skip rather than exit to allow other datasets to process

        config = configs[dataset_name]

        data = helpers.get_data(config.dataset_name)

        train_and_evaluate(data, config)
