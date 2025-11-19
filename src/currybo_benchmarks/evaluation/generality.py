import os
import itertools
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import pandas as pd

from botorch import fit_gpytorch_mll
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood

from functools import reduce

from currybo.acquisition_strategies import SequentialAcquisition, SimpleRegret
from currybo.aggregation_functions import Mean, Sigmoid, MSE, Min

from botier import AuxiliaryObjective, HierarchyScalarizationObjective

from botorch.models import SingleTaskGP

from sklearn.ensemble import RandomForestRegressor
from typing import Dict

from currybo_benchmarks.datasets import ChemistryDatasetLoader

def evaluate(cli_arg):
    pass

def find_generality(dataset: ChemistryDatasetLoader, estimated_optimum: Dict[str, str], proxy_models: Dict[str, RandomForestRegressor], cli_args: Dict):
    aggregation_function = eval(cli_args['aggregation'])(**cli_args['aggregation_kwargs'])

    x_combination = [dataset.smiles_to_fingerprint(estimated_optimum[i]) for i in estimated_optimum]
    x_combination_t = torch.cat(x_combination).unsqueeze(0)
    w_combinations = list(itertools.product(*dataset.w_options.values()))
    w_combinations_t = torch.stack([torch.cat(combo) for combo in w_combinations])

    Nx, Dx = x_combination_t.shape
    Nw, Dw = w_combinations_t.shape

    x_exp = x_combination_t[:, None, :].expand(Nx, Nw, Dx)
    w_exp = w_combinations_t[None, :, :].expand(Nx, Nw, Dw)
    points = torch.cat([x_exp, w_exp], dim=-1).reshape(Nx * Nw, Dx + Dw)

    points_np = points.numpy()  # to NumPy for sklearn

    x_arr = []
    for i in cli_args['targets']:
        name = i['name']
        model = proxy_models[name]
        pred = model.predict(points_np)            # predict all at once
        pred_t = torch.from_numpy(pred).view(Nx, Nw)
        x_arr.append(pred_t)
    x_arr_t = torch.stack(x_arr, dim=1)  # shape (num_models, Nx, Nw)
    samples = x_arr_t.unsqueeze(dim=1).unsqueeze(dim=0)

    aggregated_samples = torch.stack([
        aggregation_function(
            torch.unsqueeze(samples[:,:,:,i,:], dim=3)
            ) 
        for i in range(samples.shape[3]) 
        ], dim=3)

    # # NOTE: We go from 
    # #   num_mc_samples . n . q . m . r 
    # # to (aggregation)
    # #   num_mc_samples . n . q . m
    # # to (global objective)
    # #   num_mc_samples . n . q
    # # Shape of X: n . q . d

    # # 1 x 65 x 1 x m

    return reduce(
        lambda acc, val: { **acc, val[1]: aggregated_samples[0, 0, 0, val[0]].item() },
        [(j, i['name']) for j, i in enumerate(cli_args['targets'])],
        {},
    )

def find_max_generality(dataset: ChemistryDatasetLoader, proxy_models: Dict[str, RandomForestRegressor], cli_args: Dict):
    inputs = [i['name'] for i in cli_args['conditions']]
    outputs = [i['name'] for i in cli_args['targets']]
    objectives = [i['name'] for i in cli_args['substrates']]

    input_objectives = [i for i in objectives if i in inputs]

    aggregation_function = eval(cli_args['aggregation'])(**cli_args['aggregation_kwargs'])

    x_combinations = list(itertools.product(*dataset.x_options.values()))
    x_combinations_t = torch.stack([torch.cat(combo) for combo in x_combinations])
    w_combinations = list(itertools.product(*dataset.w_options.values()))
    w_combinations_t = torch.stack([torch.cat(combo) for combo in w_combinations])

    Nx, Dx = x_combinations_t.shape
    Nw, Dw = w_combinations_t.shape

    x_exp = x_combinations_t[:, None, :].expand(Nx, Nw, Dx)
    w_exp = w_combinations_t[None, :, :].expand(Nx, Nw, Dw)
    points = torch.cat([x_exp, w_exp], dim=-1).reshape(Nx * Nw, Dx + Dw)

    points_np = points.numpy()  # to NumPy for sklearn

    x_arr = []
    for i in cli_args['targets']:
        name = i['name']
        model = proxy_models[name]
        pred = model.predict(points_np)            # predict all at once
        pred_t = torch.from_numpy(pred).view(Nx, Nw)
        x_arr.append(pred_t)

    x_arr_t = torch.stack(x_arr, dim=1)  # shape (num_models, Nx, Nw)

    samples = x_arr_t.unsqueeze(dim=1).unsqueeze(dim=0)

    aggregated_samples = torch.stack([
        aggregation_function(
            torch.unsqueeze(samples[:,:,:,i,:], dim=3)
        ) 
        for i in range(samples.shape[3]) 
    ], dim=3)

    X = x_combinations_t.unsqueeze(dim=1)

    filename = os.path.join(cli_args['output_dir'], f"currybo-benchmark_{cli_args['dataset']}_OPTIONS.csv")
    pd.DataFrame(np.array(aggregated_samples[0, :, 0, :]), columns=[i['name'] for i in cli_args['targets']]).to_csv(filename)
    print(f":: Saved options to {filename}")

    # x_combinations = list(itertools.product(*dataset.x_options.values()))
    # x_combinations_t = [torch.cat(combo) for combo in x_combinations]
    # w_combinations = list(itertools.product(*dataset.w_options.values()))
    # w_combinations_t = [torch.cat(combo) for combo in w_combinations]
    # x_arr = []
    # for x in x_combinations_t:
    #     o_arr = []
    #     for i in cli_args['output']:
    #         name = i['name']
    #         w_arr = []
    #         for w in w_combinations_t:
    #             point = torch.concat([x, w]).unsqueeze(dim=0)
    #             w_arr.append(proxy_models[name].predict(point)[0])
    #         w_arr_t = torch.tensor(w_arr)
    #         o_arr.append(w_arr_t)
    #     o_arr_t = torch.stack(o_arr)
    #     x_arr.append(o_arr_t)
    # x_arr_t = torch.stack(x_arr) 
    # print("x_arr_t", x_arr_t.shape)

    # samples = x_arr_t.unsqueeze(dim=1).unsqueeze(dim=0)
    # print("samples", samples.shape)

    # aggregated_samples = torch.stack([
    #     aggregation_function(
    #         torch.unsqueeze(samples[:,:,:,i,:], dim=3)
    #     ) 
    #     for i in range(samples.shape[3]) 
    # ], dim=3)
    # print("aggr_samples", aggregated_samples.shape)

    # X = torch.stack(x_combinations_t).unsqueeze(dim=1)
    # print("X", X.shape)

    # NOTE: We go from 
    #   num_mc_samples . n . q . m . r 
    # to (aggregation)
    #   num_mc_samples . n . q . m
    # to (global objective)
    #   num_mc_samples . n . q
    # Shape of X: n . q . d

    # 1 x 65 x 1 x m

    # train_x: 1 x 43 x 1024
    # train_y: 1 x 43 x 2

    def get_objective(params):
        is_input_objective = params['name'] in input_objectives

        if not is_input_objective:
            params['output_index'] = outputs.index(params['name'])
        else:
            params['calculation'] = lambda y, x: x[..., 0]

        params_clean = {key: val for (key, val) in params.items() if key != 'name'}
        return AuxiliaryObjective(**params_clean)


    if not cli_args['use_botier_for_single_objective'] and len(cli_args['objectives']) == 1:
        objective_values = -aggregated_samples if cli_args['objectives'][0]['maximize'] == False else aggregated_samples
    else:        
        objectives = [get_objective(params) for params in cli_args['objectives']]
        global_objective = HierarchyScalarizationObjective(objectives, k=cli_args['smoothing_factor'], normalized_objectives=True, final_objective_idx=cli_args['final_objective'])

        # samples_to_maximize_arr = [] 
        # for m in range(aggregated_samples.shape[3]):
        #     objectives = [i for i in cli_args['objectives'] if 'output_index' in i and i['output_index'] == m]    
        #     if len(objectives) == 0:
        #         samples_to_maximize_arr.append(aggregated_samples[:, :, :, m])
        #     else:
        #         o = objectives[0]
        #         if o['maximize'] == False:
        #             samples_to_maximize_arr.append(-aggregated_samples[:, :, :, m])
        #         else:
        #             samples_to_maximize_arr.append(aggregated_samples[:, :, :, m])
        # samples_to_maximize = torch.stack(samples_to_maximize_arr, dim=3)

        # print(samples_to_maximize)

        objective_values = global_objective.forward(samples = aggregated_samples, X = X)[0,:,0]
    ma = int(torch.argmax(objective_values).item())
    # print(objective_values, ma, reduce(
    #     lambda acc, val: { **acc, val[1]: aggregated_samples[0, ma, 0, val[0]].item() },
    #     [(j, i['name']) for j, i in enumerate(cli_args['targets'])],
    #     {},
    # ), objective_values[ma])
    # sys.exit(0)

    return reduce(
        lambda acc, val: { **acc, val[1]: aggregated_samples[0, ma, 0, val[0]].item() },
        [(j, i['name']) for j, i in enumerate(cli_args['targets'])],
        {},
    )
