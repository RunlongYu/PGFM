import time
import logging
import torch
import pandas as pd
from utils.utils import combine_dataset
from utils.function_utils import log
from stage1_trainer.fm_s1_trainer import evolution_search
from config.configs import General_Config
from utils.utils import load_pd_data, set_seed
from utils.utils import USE_FEATURES
import os
import sys


def train(params):
    set_seed(params.seed)

    strategy = params.strategy

    if params.model == 'FM':
        param_save_dir = os.path.join(
            '../param/lake' + '_' + strategy, time.strftime("%Y-%m-%d-%H-%M", time.localtime()))
    elif params.model == 'FM_TF':
        param_save_dir = os.path.join(
            '../param_tf/lake' + '_' + strategy, time.strftime("%Y-%m-%d-%H-%M", time.localtime()))
        

    if not os.path.exists(param_save_dir):
        param_save_dir_fis_type = os.path.join(
            param_save_dir, "evolution", "operation_type")
        param_save_dir_alphabeta = os.path.join(
            param_save_dir, "evolution", "alpha_beta")
        os.makedirs(param_save_dir_fis_type)
        os.makedirs(param_save_dir_alphabeta)

    log(dataset=params.dataset, model=params.model, strategy=params.strategy)
    logging.info('-' * 50)
    logging.info(str(time.asctime(time.localtime(time.time()))))
    logging.info('-' * 50)
    logging.info(f'Seed:{params.seed}')
    features = USE_FEATURES
    logging.info(f"feat_size:{USE_FEATURES}")
    logging.info(f"feat_size number:{len(USE_FEATURES)}")

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:' + str(params.gpu)
    # feature_index = build_input_features(fixlen_feature_columns)

    logging.info(params)
    logging.info(General_Config)
    read_csv_extra_args = {}
    if General_Config['general']['data'] != -1:
        read_csv_extra_args = dict(nrows=General_Config['general']['data'])

    ids = pd.read_csv('../utils/intersection_ids.csv')
    ids = ids['nhdhr_id'].to_list()
    id_range = 1
    logging.info(f"id range ------ ids[:{id_range}]")
    if id_range != -1:
        ids = ids[:id_range]
    else:
        ids = ids
    train_pd_data = load_pd_data(
        ids=ids, dataset='train', save_path='../data/processedByLake/')
    test_pd_data = load_pd_data(
        ids=ids, dataset='test', save_path='../data/processedByLake/')
    val_pd_data = load_pd_data(
        ids=ids, dataset='val', save_path='../data/processedByLake/')

    stage1_data_train_do = train_pd_data['norm_do_data']
    stage1_data_train_temp = train_pd_data['temp_data']

    stage1_data_val_do = val_pd_data['norm_do_data']
    stage1_data_val_temp = val_pd_data['temp_data']

    stage1_data_test_do = test_pd_data['norm_do_data']
    stage1_data_test_temp = test_pd_data['temp_data']

    print("-----Combine Train data------")
    stage1_data_train = combine_dataset(
        stage1_data_train_do, stage1_data_train_temp)

    print("-----Combine Validatioon data------")
    stage1_data_val = combine_dataset(stage1_data_val_do, stage1_data_val_temp)

    print("-----Combine Test data------")
    stage1_data_test = combine_dataset(stage1_data_test_do, stage1_data_test_temp)

    # Pre-train use total dataset (train + val + test)
    stage1_data_train = pd.concat([stage1_data_train, stage1_data_val, stage1_data_test], ignore_index=True)
    logging.info(str(time.asctime(time.localtime(time.time()))))
    mutation = bool(params.mutation)
    evolution_search(feature_columns=features,
                     data_train=stage1_data_train, data_test=None, data_val=stage1_data_val,
                     param_save_dir=param_save_dir,
                     mutation=mutation, label_name=params.stage1_label, obs=params.stage2_label,
                     strategy=strategy,
                     embedding_size=General_Config['general']['lake_embedding_size'],
                     device=device, seed=params.seed, model_type = params.model)
