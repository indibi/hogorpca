import sys, os
from pathlib import Path
from pprint import pprint
from copy import deepcopy
from argparse import ArgumentParser, Namespace

from dask.distributed import Client, wait
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yaml
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))


from experiment_board.anomaly_detection_journal_exps.simulated_experiment_classes import Experiment, hp_study, SNN_LOGN_GTVObjective, get_data, calculate_metrics


parser = ArgumentParser(description="Specify the experiment name and the config file to repeat the experiment using best found hyper-parameters.")

parser.add_argument('group_name', type=str, 
                    help='Name of the experiment group which is also the folder name')
parser.add_argument('config_name', type=str, default='default',
                    help='Name of the yaml file configuring the experiment setting')
parser.add_argument('seed', type=int, default=0,
                    help='Start seed for the experiment')
parser.add_argument('--repeat', '-r', type=int, default=12,
                    help='Number of times to repeat the experiment with different seeds')
parser.add_argument('--n_workers', type=int, default=torch.cuda.device_count(), 
                    help='Number of workers for the experiment. Default is 4.')
parser.add_argument('--models', type=str, default=None,
                    help=('Comma-separated list of model names to run the experiment with.'
                          'Default is None, which means all models in the experiment configuration will be run.'))
parser.add_argument('--overwrite', action='store_true',
                    help=('Comma-separated list of model names to run the experiment with.'
                          'Default is to not to overwrite previous results and append to the existing results file.'))
args: Namespace = parser.parse_args()


def load_study_results(group_name, config_name, model_name):
    """
    Load study results from a CSV file.
    """
    file_path = SCRIPT_DIR / group_name / 'results' / f"{config_name}_{model_name}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Results file {file_path} does not exist.")
    
    df = pd.read_csv(file_path)
    return df


def convert_angle_to_cartesian_on_simplex(thetas):
    """Convert hyperparameters in spherical coordinates (thetas) to Cartesian coordinates on N-simplex.
    
    Args:
        thetas (np.ndarray): Angles in radians, shape (-1,N-1) where N is the number of dimensions.
    Returns:
        np.ndarray: Cartesian coordinates on N-simplex, shape (-1,N).
    """
    dim_1 = thetas.shape[0]; dim_2 = thetas.shape[1]
    thetas = np.array(thetas, dtype=np.float64)
    sin_thetas = np.concatenate([np.ones((dim_1,1)), np.sin(thetas)], axis=1)
    cos_thetas = np.concatenate([np.cos(thetas), np.ones((dim_1,1))], axis=1)
    x = (np.cumprod(sin_thetas, axis=1)*cos_thetas)**2
    return x


def set_nested_value(data, keys, value):
    """
    Sets a value in a nested dictionary given a list of keys.

    Args:
        data (dict): The dictionary to modify.
        keys (list): A list of keys representing the path to the value.
        value: The value to set.
    """
    if not keys:
        return value
    if not isinstance(data, dict):
      data = {}
    data[keys[0]] = set_nested_value(data.get(keys[0], {}), keys[1:], value)
    return data

def choose_hp_from_study_record(study_df, model_name, ind_var_name, ind_var_value,
                                    metric='SL_rel_err', hp_choice_mapping={}, seed=None):

    ind_var_value = hp_choice_mapping.get(model_name, {}).get(ind_var_value, ind_var_value)
    # if seed is not None:
    #     df_filter &= (study_df['seed'] == seed)
    row = study_df[study_df[ind_var_name] == ind_var_value]

    if row.empty:
        print(f"No data found for independent variable ({ind_var_name}) {ind_var_value}")
        return None
    if row.shape[0] > 1:
        print(f"Multiple rows found for independent variable ({ind_var_name}) {ind_var_value}")
        row = row.sort_values(by=metric).iloc[:1]
    
    thetas = np.array(
                    row[[col for col in row.columns if col.startswith('theta_') and not col.endswith('_importance')]].values
                    ).ravel()


    lda_gtvs = list(row[[col for col in row.columns if col.startswith('lda_gtv_') and not col.endswith('_importance')]].values.ravel())
    hyperparams = {'thetas': thetas, 'lda_gtvs': lda_gtvs}
    return hyperparams


if __name__ == '__main__':
    group_name = args.group_name
    config_name = args.config_name
    seed = args.seed
    repeat = args.repeat
    n_workers = args.n_workers
    overwrite = args.overwrite
    models = args.models.split(',') if args.models else None
    EXP_DIR = SCRIPT_DIR / group_name
    EXP_CFG_DIR = EXP_DIR / 'configs'
    RES_DIR = EXP_DIR / 'results'
    CFG_DIR = SCRIPT_DIR / 'configs'

    # Check if experiment directories and config files exist
    if not EXP_DIR.exists():
        raise FileNotFoundError(f'Experiment group folder {group_name} does not exist. Please create it first.')
    if not (EXP_CFG_DIR.exists() and EXP_CFG_DIR.is_dir()):
        raise FileNotFoundError(f'Config folder does not exist in experiment folder. Please create it first.')
    if not (RES_DIR.exists() and RES_DIR.is_dir()):
        raise FileNotFoundError(f'Results folder does not exist in experiment folder. Please create it first.')
    if not (EXP_CFG_DIR / f'{config_name}.yaml').exists():
        raise FileNotFoundError(f'Config file {config_name}.yaml does not exist in config folder. Please create it first.')

    # Load the experiment configuration
    with open(EXP_CFG_DIR / f'{config_name}.yaml') as f:
        exp_config = yaml.safe_load(f)
    # Load configuration for wandb API key and other settings
    with open(BASE_DIR / 'config.yaml') as f:
        main_config = yaml.safe_load(f)
    # Load default model configurations
    with open(CFG_DIR / 'models.yaml') as f:
        default_model_configs = yaml.safe_load(f)
    
    if (EXP_CFG_DIR /  f'hp_choices.yaml').exists():
        with open(EXP_CFG_DIR / f'hp_choices.yaml') as f:
            hp_choice_mapping = yaml.safe_load(f)
            hp_choice_mapping = hp_choice_mapping.get(config_name, {})
    else:
        hp_choice_mapping = {}

    if models is not None:
        models = [model.upper() for model in models]
        models_cfg_dict = exp_config.get('models', {})
        # Append default model configurations if not present in the experiment config
        for model in models:
            if model not in models_cfg_dict.keys():
                default_model_config = default_model_configs.get(model, None)
                if default_model_config is None:
                    raise ValueError(f'Model {model} is not defined in the experiment configuration or default model configurations.')
                models_cfg_dict[model] = default_model_config
        # Filter the models in the experiment configuration to only include the specified models
        exp_config['models'] = {model: models_cfg_dict[model] for model in models if model in models_cfg_dict}
    
    independent_var = exp_config['independent_var']
    model_configs = exp_config['models']
    model_keys = list(model_configs.keys())
    model_names = [model_configs[key]['name'] for key in model_keys]

    client = Client(n_workers=n_workers)

    aggregate_results = []
    for i, model_key in enumerate(model_keys):
        model_cfg = model_configs[model_key]
        model_cfg['max_iter'] = 200
        model_name = model_cfg['name']
        print(f"Loading study results for model: {model_name}")
        hp_study_df = load_study_results(group_name, config_name, model_name)
        model_aggregate_results = []

        for j in tqdm(range(len(independent_var['range'])), desc=f"Processing {model_name}"):
            hyperparams = choose_hp_from_study_record(hp_study_df, model_key, 
                                                        independent_var['keys'][-1],
                                                        independent_var['range'][j],
                                                        hp_choice_mapping=hp_choice_mapping)
            if hyperparams is None:
                print(f"No hyperparameters found for {model_name} with {independent_var['keys'][-1]} = {independent_var['range'][j]}")
                continue
            
            model_results = []
            futures = []
            num_remote = 0
            print(f"Processing model {model_name} for {independent_var['keys'][-1]} = {independent_var['range'][j]}, {repeat} times")
            for k in range(repeat):
                data_var = set_nested_value(deepcopy(exp_config['data']),
                                            independent_var['keys'], independent_var['range'][j])
                data_var['repeated'] = False
                
                model_cfg['device'] = f'cuda:{k%torch.cuda.device_count()}'
                data_var['seed'] = seed + k

                obj = SNN_LOGN_GTVObjective(data_variables=data_var,study_config={},
                                model_config=model_cfg)
                
                if num_remote == n_workers:
                    wait(futures)
                    num_remote = 0
                    ress= client.gather(futures)
                    for res in ress:
                        res[independent_var['keys'][-1]] = independent_var['range'][j]
                    model_results.extend(ress)
                    futures = []
                
                futures.append(client.submit(obj.run_with_hyperparameters, hyperparams, seed=data_var['seed']))
                num_remote += 1
            
            if futures:
                wait(futures)
                ress = client.gather(futures)
                for res in ress:
                    res[independent_var['keys'][-1]] = independent_var['range'][j]
                model_results.extend(ress)
            
            model_result_path = RES_DIR / f"{config_name}_{model_name}_results.csv"
            model_result_df = pd.DataFrame(model_results)
            if os.path.exists(model_result_path) and not overwrite:
                model_result_df.to_csv(model_result_path, mode='a', header=False, index=False)
            else:
                model_result_df.to_csv(model_result_path, mode='w', header=True, index=False)

            model_mean_result = {kk: np.mean([m[kk] for m in model_results]) for kk in model_results[0].keys()}
            model_min_result = {'min_'+ kk: np.min([m[kk] for m in model_results]) for kk in model_results[0].keys()}
            model_max_result = {'max_'+ kk: np.max([m[kk] for m in model_results]) for kk in model_results[0].keys()}
            model_std_result = {'std_'+ kk: np.std([m[kk] for m in model_results]) for kk in model_results[0].keys()}
            model_aggregate_result = {**model_mean_result, **model_min_result, **model_max_result, **model_std_result}
            model_aggregate_result[independent_var['keys'][-1]] = independent_var['range'][j]
            model_aggregate_result['model_name'] = model_name
            model_aggregate_results.append(model_aggregate_result)
        aggregate_results += model_aggregate_results

    aggregate_path = RES_DIR / f"{config_name}_aggregate_results.csv"
    aggregate_df = pd.DataFrame(aggregate_results)

    if os.path.exists(aggregate_path) and not overwrite:
        aggregate_df.to_csv(aggregate_path, mode='a', header=False, index=['model_name', independent_var['keys'][-1]])
    else:
        aggregate_df.to_csv(aggregate_path, mode='w', header=True, index=['model_name', independent_var['keys'][-1]])
    print(f"Aggregate results saved to {aggregate_path}")
    print("Experiment completed successfully.")
    client.shutdown()
    sys.exit(0)