import math
import os
import numpy as np
import json
import sys

from concurrent.futures import ProcessPoolExecutor
from threadpoolctl import threadpool_limits

from sklearn.ensemble import RandomForestRegressor
from torch import q_per_channel_axis

from currybo.cli import load_interface, parse_arguments, run_campaign as run_currybo_campaign
from currybo.utils import keyval_to_dict
from currybo.io import Dataset

from currybo_benchmarks.datasets import ChemistryDatasetLoader, presets, load_proxy_model, create_sample_dataset, pickle_if_not_exists, update_sample_dataset, get_models
from currybo_benchmarks.evaluation import find_max_generality, find_generality
from currybo_benchmarks.io import save_results

from atpbar import atpbar, find_reporter, flushing, register_reporter

from typing import Dict, Tuple, List

import multiprocessing
multiprocessing.set_start_method('fork', force=True)

def filter_args(args, interface):
    def is_core_arg(key):
        key_with_dashes = key.replace('_', '-')
        if not key_with_dashes in interface:
            return False
        return 'omit_core' not in interface[key_with_dashes] or not interface[key_with_dashes]['omit_core']
    return {key: value for (key, value) in args.items() if is_core_arg(key)}

def cli():
    interface = load_interface()
    args = parse_arguments(interface, "currybo-benchmarks", "Benchmark CLI for currybo", "omit_bm")

    args['conditions'] = presets[args['dataset']]['conditions']
    args['targets'] = presets[args['dataset']]['targets']
    args['data'] = presets[args['dataset']]['data']
    args['substrates'] = presets[args['dataset']]['substrates']
    if args['objectives']:
        args['objectives'] = [keyval_to_dict(val) for val in args['objectives']]
    else:
        args['objectives'] = presets[args['dataset']]['objectives']

    # kwargs to utilities
    args['x_utility_kwargs'] = keyval_to_dict(args['x_utility_kwargs'])
    args['w_utility_kwargs'] = keyval_to_dict(args['w_utility_kwargs'])
    args['utility_kwargs'] = keyval_to_dict(args['utility_kwargs'])
    args['surrogate_kwargs'] = keyval_to_dict(args['surrogate_kwargs'])
    args['aggregation_kwargs'] = keyval_to_dict(args['aggregation_kwargs'])


    dataset = load_dataset(args)
    pickle_if_not_exists(dataset, args)

    res, opt_est = run_parallel_campaign(dataset, args)

    save_results(res, args, "GAP")
    save_results(opt_est, args, "OPTEST")


def load_dataset(cli_args: Dict) -> ChemistryDatasetLoader:
    input_params = cli_args['conditions']
    output_params = cli_args['targets']
    w_columns = cli_args['substrates']
    samples = cli_args['samples'] if cli_args['samples'] is not None else presets[cli_args['dataset']]['num_samples']

    loader = ChemistryDatasetLoader()
    loader.load_data(cli_args['data'])
    loader.get_samples(samples, cli_args)
    loader.preprocess_data(input_params, output_params, w_columns)

    return loader

def run_parallel_campaign(dataset: ChemistryDatasetLoader, cli_args: Dict):
    proxy_models = get_models(dataset, cli_args)
    max_generality = find_max_generality(dataset, proxy_models, cli_args)

    res = [None for i in range(cli_args['jobs'])]
    opt_est = [None for i in range(cli_args['jobs'])]

    # do not use parallel features for only one worker. easier for debugging.
    if(cli_args['workers'] > 1):
        with (
            flushing(),
            ProcessPoolExecutor(
                max_workers=cli_args['workers'],
                initializer=register_reporter,
                initargs=(find_reporter(),),
            ) as executor,
        ):
            with threadpool_limits(limits=1, user_api='blas'):
                futures = [executor.submit(run_campaign, dataset=dataset, proxy_models=proxy_models, max_generality=max_generality, cli_args=cli_args, job=i) for i in range(cli_args['jobs'])]
                res = []
                opt_est = []
                for f in futures:
                    a, b = f.result()
                    res.append(a)
                    opt_est.append(b)
    else:
        for i in range(cli_args['jobs']):
            res[i], opt_est[i] = run_campaign(dataset=dataset, proxy_models=proxy_models, max_generality=max_generality, cli_args=cli_args, job=i)

    return res, opt_est

def run_campaign(dataset: ChemistryDatasetLoader, proxy_models: Dict[str, RandomForestRegressor], max_generality: Dict[str,float], cli_args: Dict, job: int):
    np.random.seed(job)

    sample_dataset = create_sample_dataset(dataset, proxy_models, cli_args, seed=job)
    interface = load_interface()
    filtered = filter_args(cli_args, interface)

    def currybo_load_dataset(cli_args):
        input_params = cli_args['conditions']
        output_params = cli_args['targets']
        w_columns = cli_args['substrates']

        ds = Dataset()

        ds.filename = "" 
        ds.dataset = sample_dataset['data']
        ds.options = sample_dataset['options']

        ds.preprocess_data(input_params, output_params, w_columns)
        return ds


    # filtered['data'] = file['data']
    # filtered['options'] = file['options']
    filtered['silent'] = True
    filtered['seed'] = job

    first_generality = {i['name']: 0 for i in cli_args['targets']}

    results = {i['name']: [] for i in cli_args['targets']}
    opt_est = {i['name']: [] for i in cli_args['targets']}

    i = 0
    iterations = math.ceil(cli_args['budget'] / cli_args['batch_size'])

    for _ in atpbar(range(iterations), name=f'Job {job+1}/{cli_args["jobs"]}'):
        ds = currybo_load_dataset(filtered)
        res = run_currybo_campaign(ds, filtered)
        generality = find_generality(dataset, res['estimated_current_optimum']['point'], proxy_models, cli_args)
        optest = res['estimated_current_optimum']['value']

        if i == 0:
            first_generality = generality
            if cli_args['verbose']:
                print(f":: [INFO] max generality:   {max_generality}")
                print(f":: [INFO] first generality: {first_generality}")

        for field in generality:
            if max_generality[field] == first_generality[field]:
                gap = np.nan
            else:
                gap = (generality[field] - first_generality[field]) / (max_generality[field] - first_generality[field])

            oe = np.sqrt((optest[field] - max_generality[field])*(optest[field] - max_generality[field]))

            results[field].append(gap)
            opt_est[field].append(oe)

        i += cli_args['batch_size']

        if i < cli_args['budget']:
            # we don't need to update the data file when it's the last iteration anyway
            sample_dataset = update_sample_dataset(dataset, sample_dataset, res, proxy_models, cli_args)

    return results, opt_est


if __name__ == "__main__":
    cli()
