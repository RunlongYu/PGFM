import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
import torch
import numpy as np
from config.temp_config import fmPgConfig
from config.configs import FM_Config, General_Config
from model.FM_Model import FM_Model
from utils.utils import load_pd_data, make_data_temp, set_seed, make_data_phys, getHypsographyManyLakes, combine_dataset
from utils.utils import USE_FEATURES, UNSUP_TEMP_FEATURES
import pickle as pkl
import argparse


def load_data(dataset, raw_dataset, label_name, lake_id):
    # bucket_data_train['datetime'] = pd.to_datetime(bucket_data_train['datetime'])
    # numeric_data_train['datetime'] = pd.to_datetime(numeric_data_train['datetime'])
    input, label, dates = make_data_temp(
        dataset, label_name, USE_FEATURES)
    phys_data, _, _ = make_data_phys(
        raw_dataset, label_name, UNSUP_TEMP_FEATURES)
    unique_depths = np.unique(dataset['vertical_depth'])
    n_depth = len(unique_depths)
    hyp_path = f"../data/processedByLake/{lake_id}/geometry"
    hypsography = getHypsographyManyLakes(hyp_path, lake_id, unique_depths)
    print(f"n depth:{n_depth}-------------- hypsography length:{hypsography.shape}")

    obs_label = label[:,:,-1]
    sim_label = label[:,:,-2]

    return input, obs_label, sim_label, phys_data, hypsography, dates



parser = argparse.ArgumentParser(description="Process model type.")
parser.add_argument("--stage2_datetime", type=str, default='')
args = parser.parse_args()
stage2_datetime = args.stage2_datetime

ids_pd = pd.read_csv(f'../utils/intersection_ids.csv')
ids = ids_pd['nhdhr_id'].to_list()
for lake_id in ids:
    seed = 40
    device = 'cpu'
    label_name = 'obs_do'
    ids = [lake_id]
    train_pd_data = load_pd_data(
        ids=ids, dataset='train', save_path='../data/processedByLake/')
    test_pd_data = load_pd_data(
        ids=ids, dataset='test', save_path='../data/processedByLake/')
    val_pd_data = load_pd_data(
        ids=ids, dataset='val', save_path='../data/processedByLake/')

    stage1_data_train_do = train_pd_data['norm_do_data']
    stage1_data_train_raw_do = train_pd_data['raw_do_data']
    stage1_data_train_temp = train_pd_data['temp_data']

    stage1_data_val_do = val_pd_data['norm_do_data']
    stage1_data_val_raw_do = val_pd_data['raw_do_data']
    stage1_data_val_temp = val_pd_data['temp_data']

    stage1_data_test_do = test_pd_data['norm_do_data']
    stage1_data_test_raw_do = test_pd_data['raw_do_data']
    stage1_data_test_temp = test_pd_data['temp_data']

    print("-----Combine Train data------")
    stage2_data_train = combine_dataset(stage1_data_train_do, stage1_data_train_temp)
    stage2_raw_data_train = combine_dataset(stage1_data_train_raw_do, stage1_data_train_temp)
    print("-----Combine Validatioon data------")
    stage2_data_val = combine_dataset(stage1_data_val_do, stage1_data_val_temp)
    stage2_raw_data_val = combine_dataset(stage1_data_val_raw_do, stage1_data_val_temp)

    print("-----Combine Test data------")
    stage2_data_test = combine_dataset(stage1_data_test_do, stage1_data_test_temp)
    stage2_raw_data_test = combine_dataset(stage1_data_test_raw_do, stage1_data_test_temp)

    trn_data, trn_label, trn_sim_label, trn_phys_data, hypsography, trn_dates = load_data(
        stage2_data_train, stage2_raw_data_train, label_name,  lake_id)
    
    val_data, val_label, val_sim_label, val_phys_data, _, val_dates = load_data(
        stage2_data_val, stage2_raw_data_val, label_name,  lake_id)
    
    test_data, test_label, test_sim_label, test_phys_data, _, test_dates = load_data(
        stage2_data_test, stage2_raw_data_test, label_name,  lake_id)


    pretrain_param_save_dir = os.path.join(
        '../param/lake' + '_' + 'n+1', stage2_datetime)

    train_config = fmPgConfig
    embedding_size = General_Config['general']['lake_embedding_size']
    selected_interaction_type = pkl.load(open(os.path.join(pretrain_param_save_dir, 'interaction_type-embedding_size-' + str(embedding_size) + '.pkl'), 'rb'))
    model = FM_Model(feature_columns=USE_FEATURES,
                            param_save_dir=pretrain_param_save_dir,
                            selected_interaction_type=selected_interaction_type,
                            mutation=0,
                            mutation_probability=0,
                            embedding_size=General_Config['general']['lake_embedding_size'],
                            device=device, seed=seed)

    ft_datetime = '2024-08-09-03-10'
    finetune_param_dir = f'../stage2_results/fm-pg/{ft_datetime}/{lake_id}/40'
    model.load_state_dict(torch.load(os.path.join(finetune_param_dir, 'model' + '.pth')))
    train_pred = model.get_temp(trn_data)
    val_pred = model.get_temp(val_data)
    test_pred = model.get_temp(test_data)

    save_path = f'../data/temp_by_model/{lake_id}/'
    os.makedirs(save_path, exist_ok=True)
    train_save_path = os.path.join(save_path, f'pred_train.npy')
    val_save_path = os.path.join(save_path, f'pred_val.npy')
    test_save_path = os.path.join(save_path, f'pred_test.npy')


    np.save(train_save_path, train_pred, allow_pickle=True)
    np.save(val_save_path, val_pred, allow_pickle=True)
    np.save(test_save_path, test_pred, allow_pickle=True)