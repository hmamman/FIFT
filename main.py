import argparse
import time
import joblib
import gc
import tensorflow.keras.backend as K
from utils.ml_classifiers import CLASSIFIERS
from utils.helpers import get_config_dict


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument('--approach_name', type=str, default='grft', required=False, help='The name of fairness testing approach to run')
    parser.add_argument('--max_allowed_time', type=int, default=3600, help='Maximum time allowed for the experiment')
    parser.add_argument('--max_iteration', type=int, default=5, help='Maximum experiment iterations')
    return parser.parse_args()

def run_approach(approach_name, config, model, classifier_name, sensitive_param, max_time=3600):
    """Run the specified fairness testing approach."""
    if approach_name == 'fift':
        from manuscript.src.fift_orignal import FIFT
        fi_threshold_type = '25_percentile'  # Options: 'std', 'percentile', 'simple', 'mean'
        fi_method = 'permutation'  # Options: 'permutation', 'shap', 'surrogate', 'uniform'
        mutation_types = ['uniform', 'dim', 'iim']

        for mutation_type in mutation_types:
            FIFT(
                config=config,
                model=model,
                sensitive_param=sensitive_param,
                fi_method=fi_method,
                fi_threshold_type=fi_threshold_type,
                mutation_type=mutation_type,
                classifier_name=classifier_name
                ).run(max_allowed_time=3600//4, max_samples=250000)
            K.clear_session()
            gc.collect()
            time.sleep(5)
    elif approach_name == 'fairmies':
        from figa_tutorial.miga import FairMIES
        FairMIES(config=config, model=model, sensitive_param=sensitive_param).run(max_allowed_time=max_time)
    elif approach_name == 'figa_ablation':
        from figa_tutorial.figa_orignal import FIGA

        for duplicate_penalty in [False]:
            FIGA(
                config=config,
                model=model,
                sensitive_param=sensitive_param,
                fi_method='shap',
                fi_threshold_type='25_percentile',
                duplicate_penalty=True,
                classifier_name=classifier_name
            ).run(max_allowed_time=3600//4, max_samples=250_000)
            K.clear_session()
            gc.collect()
            time.sleep(5)
    elif approach_name == 'adf':
        from baseline.adf.adf import ADF
        ADF(config=config, model=model, sensitive_param=sensitive_param).run(max_allowed_time=max_time)
    elif approach_name == 'eidig':
        from baseline.eidig.eidig import EIDIG
        EIDIG(config=config, model=model, sensitive_param=sensitive_param).run(max_allowed_time=max_time)
    elif approach_name == 'maft':
        from baseline.maft.maft import MAFT
        MAFT(config=config, model=model, sensitive_param=sensitive_param).run(max_allowed_time=max_time)
    elif approach_name == 'aequitas':
        from baseline.aequitas.aequitas import Aequitas
        Aequitas(config=config, model=model, sensitive_param=sensitive_param).run(max_allowed_time=max_time)
    elif approach_name == 'expga':
        from baseline.expga.expga import ExpGA
        ExpGA(config=config, model=model, sensitive_param=sensitive_param).run(max_allowed_time=max_time)
    elif approach_name == 'grft':
        from baseline.grft.grft import GRFT 
        GRFT(config=config, model=model, sensitive_params=[sensitive_param], classifier_name=classifier_name).run(max_allowed_time=max_time)

args = parse_arguments()
approaches = [
    'aequitas', 'expga', 'figa', 'fift', 'fairmies', 'figa_ablation', 'adf', 'neuron_fair', 'eidig', 'maft', 'grft'
]

dnn_based = ['adf', 'neuron_fair', 'eidig', 'maft']

if args.approach_name not in approaches:
    raise ValueError(f"Invalid sensitive name: {args.approach_name}. Available options are: {approaches}")

for _ in range(args.max_iteration):
    for config in get_config_dict().values():
        # classifier_names = list(CLASSIFIERS.keys()) + ['dnn'] if args.approach_name not in dnn_based else ['dnn']
        classifier_names = ['lr', 'dt', 'rf', 'dnn']
        for classifier_name in classifier_names:
            for sensitive_param in config.sens_name:
                print(f'Approach name: {args.approach_name}')
                print(f'Dataset: {config.dataset_name}')
                print(f'Classifier: {classifier_name}')
                print(f'Sensitive name: {config.sens_name[sensitive_param]} {sensitive_param}')

                if classifier_name == 'dnn':
                    import tensorflow as tf
                    classifier_path = f'models/{config.dataset_name}/{classifier_name}.keras'
                    model = tf.keras.models.load_model(classifier_path)
                else:
                    classifier_path = f'models/{config.dataset_name}/{classifier_name}.pkl'
                    model = joblib.load(classifier_path)

                run_approach(
                    approach_name=args.approach_name,
                    config=config,
                    model=model,
                    sensitive_param=sensitive_param,
                    max_time=args.max_allowed_time,
                    classifier_name=classifier_name
                )
                K.clear_session()
                gc.collect()