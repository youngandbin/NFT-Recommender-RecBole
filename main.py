import torch
import pandas as pd
import numpy as np
import scipy.sparse
import yaml
from tqdm import tqdm
import shutil
import os
import glob
import wandb
import argparse

from logging import getLogger
from recbole.quick_start import run_recbole
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from recbole.utils import get_model, get_trainer
from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function

# random seed
SEED = 2022
np.random.seed(SEED)
torch.manual_seed(SEED)

"""
arg parser
"""
parser = argparse.ArgumentParser(description='recbole baseline')
parser.add_argument('--model', type=str, default='BPR')
parser.add_argument('--dataset', type=str, default='bayc')
parser.add_argument('--item_cut', type=int, default=3)
parser.add_argument('--config', type=str, default='baseline')
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

"""
arg parser -> variables
"""
MODEL = args.model
DATASET = args.dataset
ITEM_CUT = args.item_cut
CONFIG = f'config/fixed_config_{args.config}.yaml'

"""
main functions
"""

def objective_function(config_dict=None, config_file_list=None):
    
    config = Config(model=MODEL, dataset=DATASET, config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model_name = config['model']
    model = get_model(model_name)(config, train_data.dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    """ (1) training """
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False)
    """ (2) testing """
    test_result = trainer.evaluate(test_data)

    return {
        'model': model_name,
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }

def main_HPO():

    hp = HyperTuning(objective_function=objective_function, algo="exhaustive",
                     max_evals=50, params_file=f'hyper/{MODEL}.hyper', fixed_config_file_list=[f'config/fixed_config_baseline.yaml'])

    # run
    hp.run()
    # export result to the file
    hp.export_result(
        output_file=f'hyper_result/{MODEL}_{DATASET}.result')
    # print best parameters
    print('best params: ', hp.best_params)
    # save best parameters
    with open(f'hyper_result/{MODEL}_{DATASET}.best_params', 'w') as file:
        documents = yaml.dump(hp.best_params, file)
    # print best result
    best_result = hp.params2result[hp.params2str(hp.best_params)]

    best_result_df = pd.DataFrame.from_dict(
        best_result['test_result'], orient='index', columns=[f'{DATASET}'])
    best_result_df.to_csv(
        result_path + f'{MODEL}-{DATASET}.csv', index=True)

def main():
    
    config = Config(model=MODEL, dataset=DATASET, config_file_list=[CONFIG])
    
    # init random seed
    init_seed(config['seed'], config['reproducibility'])

    # dataset creating and filtering # convert atomic files -> Dataset
    dataset = create_dataset(config)

    # dataset splitting # convert Dataset -> Dataloader
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # model loading and initialization
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    
    """ (1) training """
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    """ (2) testing """
    trainer.eval_collector.data_collect(train_data)
    test_result = trainer.evaluate(test_data)
    # save result
    result_df = pd.DataFrame.from_dict(test_result, orient='index', columns=[f'{DATASET}'])
    result_df.to_csv(result_path + f'{MODEL}-{DATASET}.csv', index=True)


"""
main
"""
if __name__ == '__main__':
    
    # wandb
    wandb.init(project="nft-recommender", name=f'{MODEL}_{DATASET}', entity="nft-recommender")
    wandb.config.update(args)
    
    # result path
    result_path = './result/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    main()
    
