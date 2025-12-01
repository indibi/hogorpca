import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
import logging
import datetime
import pprint
import time

import yaml

parser = ArgumentParser(description="Specify the experiment name and the config file to run simulated experiment.")

parser.usage = '''First, create a folder with the name of the experiment group.
The folder should contain the following directories:
1. configs: This folder should contain the yaml files for the experiment settings.
2. results: The results of the experiment will be saved to this folder.

Then run the script by specifying the following arguments:
python simulated_experiment_runner.py <group_name> <config_name> <seed> --repeat <repeat> --message <message>
'''

parser.add_argument('group_name', type=str, 
                    help='Name of the experiment group which is also the folder name')
parser.add_argument('config_name', type=str, default='default',
                    help='Name of the yaml file configuring the experiment setting')
parser.add_argument('seed', type=int, default=0,
                    help='Start seed for the experiment')
parser.add_argument('--repeat', '-r', type=int, default=1,
                    help='Number of times to repeat the experiment with different seeds')
parser.add_argument('-m', '--message', type=str, default='',
                    help='Message to be printed in the experiment log file and appended to the experiment notes.')
parser.add_argument('-s', '--store-hp-study', action='store_true', 
                    help='Whether to store the hyperparameter study results in the database. Default is False.')
parser.add_argument('--n_workers', type=int, default=4, 
                    help='Number of workers for the experiment. Default is 4.')
parser.add_argument('--models', type=str, default=None,
                    help=('Comma-separated list of model names to run the experiment with.'
                          'Default is None, which means all models in the experiment configuration will be run.'))
args: Namespace = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from experiment_board.anomaly_detection_journal_exps.simulated_experiment_classes import Experiment


if __name__ == '__main__':
    group_name = args.group_name
    config_name = args.config_name
    seed = args.seed
    repeat = args.repeat
    message = args.message
    store_hp_study = args.store_hp_study
    n_workers = args.n_workers
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


    # Set up the experiment configuration
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


    wandb_api_key = main_config['wandb']['api_key']
    if store_hp_study:
        exp_config['study_config']['storage'] = f'sqlite:///{RES_DIR.absolute()}/{config_name}_optuna_studies.db'
    else:
        exp_config['study_config']['storage'] = None
    exp_config['result_csv_path'] = str(RES_DIR.resolve())#f'{RES_DIR.absolute()}/{config_name}_results.csv'
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=(EXP_DIR/'journal.log').absolute(),
                        encoding='utf-8',
                        filemode='a',
                        level=logging.INFO,
                        )
    
    
    start_time = time.time()
    start_info_log = f'''
    --------------------------------------
    Initiating experiment. Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    Experiment group name: {group_name}
    Config name: {config_name}
    Seed: {seed}
    Repeat: {repeat}
    Message: {message}
    Models: {exp_config['models'].keys()}
    exp_config: {pprint.pformat(exp_config)}
    '''
    logger.info(start_info_log)
    exp = Experiment(api_key=wandb_api_key,
                     exp_config=exp_config,
                     n_workers=n_workers)
    
    exp.run_experiment(seed=seed, n_trials=repeat, message=message)
    end_time = time.time()
    elapsed_time = end_time - start_time
    end_info_log = f'''
    Experiment finished. Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    Elapsed time: {elapsed_time/60:.2f} minutes.
    --------------------------------------
    '''
    logger.info(end_info_log)
