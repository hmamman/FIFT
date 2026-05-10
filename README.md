# FIFT: Feature Importance-Guided Fairness Testing for Machine Learning Software

## Installation

1. Download/Clone the repository:
   ```bash
   Download from: https://anonymous.4open.science/r/FIFT/
   Unzip and cd into the directory


2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
## Datasets and Protected Attributes
| Dataset | Protected Attribute | Index (Starts at 1) |
|---------|---------------------|---------------------|
| census  | sex                 | 9                   |
|         | age                 | 1                   |
|         | race                | 8                   |
| credit  | sex                 | 9                   |
|         | age                 | 13                  |
| bank    | age                 | 1                   |
|         | marital             | 3                   |
| compas  | sex                 | 1                   |
|         | age                 | 2                   |
|         | race                | 3                   |
| meps    | sex                 | 3                   |

Dataset and protected attribute names are case-sensitive.

## Running Fairness Testing

### Command-Line Arguments

The script accepts the following arguments:

- `--dataset_name`: (string) Name of the dataset to use in the experiment. The default is `'census'`.
  - Example: `--dataset_name census`

- `--sensitive_name`: (string) Name of the protected attribute for fairness testing (e.g., `sex`, `age`, `race`). The default is `'age'`.
  - Example: `--sensitive_name sex`

- `--classifier_name`: (string) Name of the classifier to use (e.g., `mlp`, `dt`, `rf`, ect.). The default is `'dt'`.
  - Example: `--classifier_name svm`

- `--max_allowed_time`: (integer) Maximum time in seconds for the experiment to run. The default is `3600` seconds (1 hour).
  - Example: `--max_allowed_time 3600`

### Example Usage

To run the FIGA framework:
```bash
python ./figa_tutorial/figa.py --classifier_name dt --dataset_name bank --sensitive_name age --max_allowed_time 3600
```

To run a specific benchmarking approach included in this repository:
```bash
python ./baseline/expga/expga.py --classifier_name dt --dataset_name bank --sensitive_name age --max_allowed_time 3600
```

You can also run all experiments for an approach (figa, expga, sg, and fairbs) using main.py file:
```bash
python ./main.py --approach_name fairbs --max_allowed_time 3600 --max_iteration 5  
```

