import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from copy import deepcopy
import logging
import datetime
from pprint import pprint
import time

import yaml
from dask.distributed import Client, as_completed
import torch

parser = ArgumentParser(description="Specify the experiment name and the config file to run Server Machine Dataset Anomaly Detection Algorithm.")

parser.usage = '''First, create a folder with the name of the experiment group.
The folder should contain the following directories:
1. configs: This folder should contain the yaml files for the experiment settings.
2. results: The results of the experiment will be saved to this folder.

Then run the script by specifying the following arguments:
python smd_experiment_runner.py 'model1,model2,model3,model4' <machine_id> <channel_id> <metric>
'''

parser.add_argument('models', type=str, 
                    help='Names of the models to run the experiment on. This should be a comma-separated list of model names. For example: "HoRPCA_f,LR-STS_f"')
parser.add_argument('--machine_id',
                    type=int, default=1,
                    help=('machine id to run the experiment on. Default is 1. Valid values are 0, 1, 2, 3.',
                          '0 loads New York City Taxi dataset'))
parser.add_argument('--channel_id',
                    type=int, default=1,
                    help='channel id to run the experiment on. Default is 1. Valid values are 1 to 8 for machine 1, 1 to 9 for machine 2, and 1 to 11 for machine 3.')
parser.add_argument('--metric', type=str, default='gic_5',
                    help='Metric to minimize during the hyperparameter optimization. Default is "gic_5". Valid values are "gic_[1-6]", "dof", "bic" and more.')
parser.add_argument('--num_trials', type=int, default=300,
                    help='Number of trials to run for the hyperparameter optimization. Default is 100')
# parser.add_argument('--max_iter', type=int, default=2000,
#                     help='Maximum number of iterations for model convergence. Default is 2000')
# parser.add_argument('--err_tol', type=float, default=0.001,
#                     help='Maximum number of iterations for model convergence. Default is 2000')
parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite the existing hyperparameter study. Default is False.')
parser.add_argument('--append', action='store_true',
                    help='Whether to append n_trials trials to the existing studies or complete the number of trials to n_trials . Default is False.')


args: Namespace = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / 'results'
CFG_DIR = SCRIPT_DIR / 'configs'
sys.path.append(BASE_DIR.as_posix())

from experiment_board.smd_anomaly_detection.smd_hp_study_classes import StudyWrapper


if __name__ == '__main__':
    models = args.models.split(',')
    machine_id = args.machine_id
    channel_id = args.channel_id
    metric = args.metric
    overwrite = args.overwrite
    # max_iter = args.max_iter
    num_trials = args.num_trials
    append_or_complete = 'append' if args.append else 'complete'
    
    
    cfg_customization = {
        'model':{},#'max_iter': max_iter},
        'study':{
            'overwrite': overwrite,
        }
    }
    client = Client(n_workers=torch.cuda.device_count()+1)
    print(client)
    studies = []
    for i,model in enumerate(models):
        cfg = deepcopy(cfg_customization)
        cfg['model']['device'] = f'cuda:{i%torch.cuda.device_count()}'
        # study_actor = client.submit(StudyWrapper, model, 
        #                     machine_id, channel_id,
        #                     metric, cfg_customization=cfg, actor=True)
        # studies.append(study_actor.result())
        studies.append(StudyWrapper(model, 
                            machine_id, channel_id,
                            metric, cfg_customization=cfg))
    
    # study_refs = []
    # for i, study_actor in enumerate(studies):
    #     future = study_actor.run_study(n_trials=num_trials,
    #                                     device=f'cuda:{i%torch.cuda.device_count()}',
    #                                     append_or_complete=append_or_complete)
    #     study_refs.append(future)
    
    # for future in as_completed(study_refs):
    #     result = future.result()
    #     print("Study completed with result:"+"\n"+"-"*20)
    #     pprint(result)
        


    # pprint(models)
    study_refs = []
    for i, study in enumerate(studies):
        study_refs.append(client.submit(study.run_study,
                                        n_trials=num_trials,
                                        device=f'cuda:{i}',
                                        append_or_complete=append_or_complete))
    
    for future in as_completed(study_refs):
        result = future.result()
        print("Study completed with result:"+"\n"+"-"*20)
        pprint(result)