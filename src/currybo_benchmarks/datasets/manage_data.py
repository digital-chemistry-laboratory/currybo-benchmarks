import numpy as np
import os
import pandas as pd
import pickle
import tempfile
import sys

from datetime import datetime
from platformdirs import user_cache_dir
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from .discrete_problem import ChemistryDatasetLoader

from typing import Dict, List

def get_data_file(dataset_name: str, timestamp: str, seed: int, kind: str):
    tmpdir = tempfile.gettempdir()
    return os.path.join(tmpdir, f"currybo-benchmark_{kind}_{dataset_name}_{timestamp}_sample-{seed}.csv")

def create_sample_dataset(dataset, proxy_models: Dict[str, RandomForestRegressor], cli_args: Dict, seed=1):
    full = dataset.dataset
    initial_seed = cli_args['initial_seed']
    objectives = [i['name'] for i in cli_args['targets'] if i not in cli_args['conditions']]
    options_cols = [i for i in full.columns if i not in objectives]

    data = full.sample(n = initial_seed, random_state=seed)
    options = full[options_cols].reset_index()

    data_to_save = []
    for _, row in data.iterrows():
        new_data = {}
        input_data = {}
        for c in [i['name'] for i in [*cli_args['conditions'], *cli_args['substrates']]]:
            input_data[c] = row[c]
        for c in row.keys():
            new_data[c] = row[c]
        for o in objectives:
            fp = convert_cols_to_fingerprint(dataset, pd.DataFrame([input_data]), cli_args)
            true_value = proxy_models[o].predict(fp)[0]
            new_data[o] = true_value
        data_to_save.append(new_data)
    D = pd.DataFrame(data_to_save).reset_index(drop=True)

    # d = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")

    #data_file = get_data_file(dataset.dataset_name, d, seed, "data")
    #options_file = get_data_file(dataset.dataset_name, d, seed, "options")
    #D.to_csv(data_file)
    #options.to_csv(options_file)

    # if cli_args['verbose']:
    #     print(f":: [INFO] Data File for sample {seed}:    {data_file}")
    #     print(f":: [INFO] Options File for sample {seed}: {options_file}")

    return {
        "data": D,
        "options": options
    }

def update_sample_dataset(dataset: ChemistryDatasetLoader, sample_dataset: Dict, results: Dict, proxy_models: Dict[str, RandomForestRegressor], cli_args: Dict):
    current_data = sample_dataset['data']
    to_append = []
    for p in results['next_points']:
        new_data = {}
        input_data = {}
        for c in [i['name'] for i in [*cli_args['conditions'], *cli_args['substrates']]]:
            input_data[c] = p['point'][c]
        for c in p['point']:
            new_data[c] = p['point'][c]
        for o in p['value']:
            fp = convert_cols_to_fingerprint(dataset, pd.DataFrame([input_data]), cli_args)
            true_value = proxy_models[o].predict(fp)[0]
            new_data[o] = true_value
        to_append.append(new_data)
    sample_dataset['data'] = pd.concat([current_data, pd.DataFrame(to_append)]).reset_index(drop=True)
    #updated_data.to_csv(files['data'])
    return sample_dataset
     

def get_pickle_file_path(dataset_name: str, objective: str):
    APPNAME = "currybo_benchmarks"
    APPAUTHOR = "currybo"

    filename = f"{dataset_name}_{objective}.pkl"
    return os.path.join(user_cache_dir(APPNAME, APPAUTHOR), filename)

def convert_cols_to_fingerprint(dataset: ChemistryDatasetLoader, data: pd.DataFrame, cli_args: Dict):
    fingerprints = []
    for _, row in data.iterrows():
        inputs = []
        for i, t in [(i['name'], i['type']) for i in [*cli_args['conditions'], *cli_args['substrates']]]:
            if t == "smiles":
                fp = dataset.smiles_to_fingerprint(row[i])
            elif t == "array":
                fp = [float(j) for j in row[i].split(" ")]
            else:
                fp = [row[i]]
            inputs.append(fp)
        combined_fp = np.concatenate(inputs)
        fingerprints.append(combined_fp)

    return np.array(fingerprints)
 
def pickle_if_not_exists(dataset, cli_args):
    objectives = [i['name'] for i in cli_args['targets'] if i not in cli_args['conditions']]

    does_pickle_file_exist = lambda o: os.path.isfile(get_pickle_file_path(dataset.dataset_name, o))

    todo = []
    for o in objectives:
        if does_pickle_file_exist(o):
            if cli_args['verbose']:
                print(f":: [INFO] Found pickle file {get_pickle_file_path(dataset.dataset_name, o)}")
        else:
            todo.append(o)
    if len(todo) == 0:
        return

    print(f":: [INFO] The pickle files for objectives [{', '.join(todo)}] do not exist. We'll create them now and cache them for later. This might take a while.")
    X = convert_cols_to_fingerprint(dataset, dataset.dataset, cli_args)
    Y = [dataset.dataset[y] for y in todo]

    for name, y in zip(todo, Y):
        model = RandomForestRegressor(n_estimators=1000, random_state=12)
        model.fit(X, y)

        # Evaluate the model
        y_pred = model.predict(X)
        mse = root_mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)

        path = get_pickle_file_path(dataset.dataset_name, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the trained model
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f":: [INFO] Pickle file for {name} created. RMSE: {mse}, MAE: {mae}. Pickle file path: {path}")

def get_models(dataset: ChemistryDatasetLoader, cli_args: Dict):
    objectives = [i['name'] for i in cli_args['targets'] if i not in cli_args['conditions']]

    models = {}
    for o in objectives:
        models[o] = dataset.load_model(get_pickle_file_path(dataset.dataset_name, o))
    return models
