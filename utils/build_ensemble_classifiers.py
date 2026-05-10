import argparse
import sys
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Get the absolute path to the directory where main.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Two levels up from expga.py
sys.path.append(os.path.join(base_path, "../"))

from utils.helpers import build_model_arguments

from utils import helpers
from utils.ml_classifiers import ensemble_pipeline

def train_and_evaluate(data, dataset_name):
    X, y, input_shape, nb_classes = data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ensemble_pipeline.fit(X_train, y_train)

    y_pred = ensemble_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"VEC Model's accuracy on {dataset_name} dataset is: {accuracy:.4f}")

    # Save the model to disk
    dir_path = f"models/ensemble"
    os.makedirs(dir_path, exist_ok=True)
    model_path = f'{dir_path}/{dataset_name}_ensemble.pkl'
    joblib.dump(ensemble_pipeline, model_path)
    print(f"Ensemble Model saved to {model_path}")
    #           Bank    Credit  Census  Compas  Meps
    #  Ensemble 0.8989  0.7667  0.8422  0.8214  0.8692


def main():

    args = build_model_arguments()

    configs = helpers.get_config_dict()

    dataset_name = args.dataset_name.lower()

    data_path = f"data/{dataset_name}.py"
    if not os.path.exists(data_path):
        print(f"Dataset not found: {data_path}")
        sys.exit(1)

    config = configs[dataset_name]

    data = helpers.get_data(config.dataset_name)

    train_and_evaluate(data, dataset_name)


if __name__ == "__main__":
    main()
