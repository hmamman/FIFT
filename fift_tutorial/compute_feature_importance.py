import numpy as np
from sklearn import cluster
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
import shap

def compute_pfi(model, X, y, n_repeats=5, random_state=42):
    res = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    return np.maximum(0.0, res.importances_mean.astype(float))

def compute_shap(model, X, X_val, random_state=42):
    # Decide on best SHAP explainer for this model
    if hasattr(model, "predict_proba"):
        predict_fn = model.predict_proba
    else:
        predict_fn = model.predict

    # Choose background dataset
    rng = np.random.RandomState(random_state)
    bg_idx = rng.choice(len(X), size=min(50, len(X)), replace=False)
    background = X[bg_idx]

    # Select the most efficient SHAP explainer
    if "Tree" in str(type(model)) or "Forest" in str(type(model)) or "Boosting" in str(type(model)):
        explainer = shap.TreeExplainer(model)
    elif "tensorflow" in str(type(model)).lower() or "keras" in str(type(model)).lower():
        explainer = shap.DeepExplainer(model, background)
    else:
        explainer = shap.KernelExplainer(predict_fn, background)

    shap_vals = explainer.shap_values(X_val[:min(100, len(X_val))])

    # Handle binary/multiclass outputs
    if isinstance(shap_vals, list):
        vals = np.mean([np.abs(v).mean(axis=0) for v in shap_vals], axis=0)
    else:
        vals = np.abs(shap_vals).mean(axis=0)

    # Collapse class dimension if needed
    if vals.ndim > 1:
        vals = np.mean(vals, axis=1)

    return vals.astype(float)

def compute_surrogate(model, X_val, random_state=42):
    # Train decision tree surrogate on model predictions
    y_pred = model.predict(X_val)
    is_class = np.issubdtype(y_pred.dtype, np.integer) or len(np.unique(y_pred)) <= 20
    # Surrogate = DecisionTreeClassifier if is_class else DecisionTreeRegressor
    surrogate = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=random_state)
    surrogate.fit(X_val, y_pred)
    return surrogate

def clustering(data, c_num=4):
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
        return x[index]

def generate_seeds(X, c_num=4, num_seeds=1000, fashion='RoundRobin'):
    clustered_data = clustering(X, c_num)
    seeds = []
    for i in range(100000000):
        x_seed = get_seed(clustered_data, len(X), c_num, i % c_num, fashion=fashion)
        seeds.append(x_seed)
        if len(seeds) >= num_seeds:
            break
    return np.array(seeds)


def compute_feature_importance(
    method: str,
    model,
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    n_repeats: int = 5,
) -> np.ndarray:


    rng = np.random.RandomState(random_state)
    n_features = X.shape[1]
    method = method.lower()

    # Train/validation split
    if len(X) > 1000:
        _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)
    else:
        X_val, y_val = X, y

    if method == "uniform":
        fi = np.ones(n_features, dtype=float)

    elif method == "permutation":

        fi = compute_pfi(
            model=model,
            X=X_val,
            y=y_val,
            n_repeats=n_repeats,
            random_state=random_state
        )

    elif method == "shap":
        fi = compute_shap(
        model=model,
        X=X,
        X_val=X_val,
        random_state=random_state
    )
    elif method == "surrogate":
        surrogate = compute_surrogate(
            model=model,
            X_val=X_val,
            random_state=random_state
        )
        fi = np.array(surrogate.feature_importances_, dtype=float)
    elif method == "surrogate_shap":
        surrogate = compute_surrogate(
            model=model,
            X_val=X_val,
            random_state=random_state
        )
        explainer = shap.TreeExplainer(surrogate)

        seeds = generate_seeds(X=X, num_seeds=1000, fashion='Distribution')

        shap_vals = explainer.shap_values(seeds)

        # Handle binary/multiclass outputs
        if isinstance(shap_vals, list):
            vals = np.mean([np.abs(v).mean(axis=0) for v in shap_vals], axis=0)
        else:
            vals = np.abs(shap_vals).mean(axis=0)

        # Collapse class dimension if needed
        if vals.ndim > 1:
            vals = np.mean(vals, axis=1)

        fi = vals.astype(float)

    else:
        raise ValueError(f"Unknown FI method: {method}")

    # # Normalize to [0,1]
    # m = float(np.max(fi)) if fi.size > 0 else 0
    # return fi / m if m > 0 else fi
    return fi