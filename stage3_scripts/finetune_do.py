import os
import sys
import time
import torch
import logging
import numpy as np
import pandas as pd
import pickle as pkl
from config.configs import FM_Config, General_Config
from config.do_config import fmPgConfig, fmLSTMConfig, fmPgPlusConfig, lstmConfig, transformerConfig, fmTFConfig, ealstmConfig
from utils.function_utils import build_input_features
from utils.utils import USE_FEATURES, UNSUP_DO_FEATURES, STATIC_FEATURES
from utils.utils import create_path, make_data, getHypsographyManyLakes
from utils.utils import start_date_list, end_date_list
from utils.utils import combine_dataset
from model.baseline_lstm import LSTM
from model.baseline_ealstm import MyEALSTM
from model.baseline_transformer import TransformerModel
from model.FM_Model import FM_Model
from model.FM_TF import FM_TF
from utils.utils import load_pd_data, make_data_do, set_seed, make_data_phys
from utils.utils_stage2 import getPbDoResults

def save_data(path, data):
    np.save(path, data)
    print(f'Data saved to {path}')

def load_data(dataset, raw_dataset, label_name, lake_id, data_type, model_type):
    # bucket_data_train['datetime'] = pd.to_datetime(bucket_data_train['datetime'])
    # numeric_data_train['datetime'] = pd.to_datetime(numeric_data_train['datetime'])

    phys_data, _, _ = make_data_phys(
        raw_dataset, label_name, UNSUP_DO_FEATURES)
    
    # temp_pred_trn = np.load(f'../stage2_results/fm-pg/2024-08-07-02-15/{lake_id}/40/pred_{data_type}.npy')

    unique_depths = np.unique(dataset['vertical_depth'])
    n_depth = len(unique_depths)
    hyp_path = f"../data/processedByLake/{lake_id}/geometry"
    hypsography = getHypsographyManyLakes(hyp_path, lake_id, unique_depths)
    print("hypsography:", hypsography)
    print("n_depth:", n_depth)
    depth_areas = hypsography.flatten()
    
    input, label, dates = make_data_do(
        dataset, label_name, USE_FEATURES, depth_areas, model_type, data_type)
    
    obs_label = label[:,:,-1]
    sim_label = label[:,:,-2]
    print(f"n depth:{n_depth}-------------- hypsography length:{hypsography.shape}")
    return input, obs_label, sim_label, phys_data, hypsography, dates

def train(args):
    strategy = args.strategy  # fix
    seed = args.seed
    lake_id = args.lake_id
    model_type = args.model_type
    pt_model_dt = args.pt_datetime
    label_name = args.label_name

    current_time = args.current_time
    set_seed(seed=seed)
    if torch.cuda.is_available() and args.gpu < torch.cuda.device_count():
        device = torch.device(f'cuda:{args.gpu}')
        print(f'Using GPU: cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    # Create path
    save_path = create_path(
        f'../stage3_results/{model_type}/{current_time}/{lake_id}/{seed}/')
    
    # Init log
    filename = str(model_type) + '_' + strategy + '_' + \
        '_result_' + str(time.strftime("%H-%M-%S", time.localtime())) +f'_seed_{seed}'+ '.log'
    logging.basicConfig(filename=os.path.join(
        f'../stage3_results/{model_type}/{current_time}/', filename), level=logging.INFO, filemode='w')

    if model_type == 'fm-transformer':
        param_save_dir = os.path.join(
            '../param_tf/lake' + '_' + strategy, pt_model_dt)
    else:
        param_save_dir = os.path.join(
            '../param/lake' + '_' + strategy, pt_model_dt)
        
    if not os.path.exists(param_save_dir):
        logging.info("------Load feature False, load_path not exist------")
        raise Exception("Load path missing")

    # LOAD DATA
    # Temperature: intpu ----> [all_batch, seq_length, features], label ----> [all_batch, seq_length]
    logging.info('-'*50)
    logging.info(f"Lake id: {lake_id}")
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
        stage2_data_train, stage2_raw_data_train, label_name,  lake_id, data_type='train', model_type = model_type)
    
    val_data, val_label, val_sim_label, val_phys_data, _, val_dates = load_data(
        stage2_data_val, stage2_raw_data_val, label_name,  lake_id, data_type='val', model_type = model_type)
    
    test_data, test_label, test_sim_label, test_phys_data, _, test_dates = load_data(
        stage2_data_test, stage2_raw_data_test, label_name,  lake_id, data_type='test', model_type = model_type)
    

    # print("hypsography:", hypsography)
    # sim_label = 'sim_temp'
    # _, trn_sim_label, _, _, _ = load_data(stage2_data_train, stage2_raw_data_train, sim_label,  lake_id, data_type='train', model_type = model_type)
    # _, val_sim_label, _, _, _ = load_data(stage2_data_val, stage2_raw_data_val, sim_label,  lake_id, data_type='val', model_type = model_type)
    # _, test_sim_label, _, _, _ = load_data(stage2_data_test, stage2_raw_data_test, sim_label,  lake_id, data_type='test', model_type = model_type)

    all_data = np.concatenate((trn_data, val_data, test_data), axis=0)
    all_phys_data = np.concatenate(
        (trn_phys_data, val_phys_data, test_phys_data), axis=0)
    all_dates = np.concatenate((trn_dates, val_dates, test_dates), axis=0)

    # Load params of pretrained model
    logging.info('\nModel Functioning period param:')
    logging.info(FM_Config['ModelFunctioning'])
    feature_names = USE_FEATURES

    embedding_size = General_Config['general']['lake_embedding_size']
    selected_interaction_type = pkl.load(open(os.path.join(param_save_dir, 'interaction_type-embedding_size-' + str(
        embedding_size) + '.pkl'), 'rb'))
    logging.info(selected_interaction_type)
    
    # Train
    # load model
    # choices=['lstm', 'ealstm', 'transformer', 'fm-lstm', 'fm-ealstm', 'fm-transformer', 'fm-pg']
    logging.info(
        "#################### Model Finetuning period start ####################")
    logging.info(f"-----Model type: {model_type}-----")

    if model_type == 'lstm':
        train_config = lstmConfig
        model = LSTM(input_size=len(feature_names), hidden_size=train_config['hidden_size'], batch_size = train_config['batch_size'], device=device)

    elif model_type == 'fm-lstm':
        train_config = fmLSTMConfig
        model = FM_Model(feature_columns=feature_names,
                                param_save_dir=param_save_dir,
                                selected_interaction_type=selected_interaction_type,
                                mutation=0,
                                mutation_probability=0,
                                embedding_size=General_Config['general']['lake_embedding_size'],
                                device=device, seed=seed)
        model.load_state_dict(torch.load(os.path.join(param_save_dir, 'model' + '.pth')))
    elif model_type == 'fm-pg':
        train_config = fmPgConfig
        model = FM_Model(feature_columns=feature_names,
                                param_save_dir=param_save_dir,
                                selected_interaction_type=selected_interaction_type,
                                mutation=0,
                                mutation_probability=0,
                                embedding_size=General_Config['general']['lake_embedding_size'],
                                device=device, seed=seed)

        model.load_state_dict(torch.load(os.path.join(param_save_dir, 'model' + '.pth')))
    elif model_type == 'fm-pg+':
        train_config = fmPgPlusConfig
        model = FM_Model(feature_columns=feature_names,
                                param_save_dir=param_save_dir,
                                selected_interaction_type=selected_interaction_type,
                                mutation=0,
                                mutation_probability=0,
                                embedding_size=General_Config['general']['lake_embedding_size'],
                                device=device, seed=seed)

        model.load_state_dict(torch.load(os.path.join(param_save_dir, 'model' + '.pth')))
    elif model_type == 'ealstm':
        train_config = ealstmConfig
        model = MyEALSTM(input_size_dyn=len(USE_FEATURES)-len(STATIC_FEATURES),
                        input_size_stat=len(STATIC_FEATURES),
                        hidden_size=train_config['hidden_size'],
                        initial_forget_bias= 5,
                        dropout= 0.4,
                        concat_static=False,
                        no_static=False, device=device)
    elif model_type == 'transformer':
        train_config = transformerConfig
        model = TransformerModel(input_size=len(feature_names), num_heads = train_config['num_heads'], num_layers=train_config['num_layers'], hidden_size=train_config['hidden_size'], device=device)
    elif model_type == 'fm-transformer':
        train_config = fmTFConfig
        model = FM_TF(feature_columns=feature_names,
                                param_save_dir=param_save_dir,
                                selected_interaction_type=selected_interaction_type,
                                mutation=0,
                                mutation_probability=0,
                                embedding_size=embedding_size, seed=seed,
                                device=device)
        model.load_state_dict(torch.load(os.path.join(param_save_dir, 'model' + '.pth')))
    elif model_type == 'PB':
        train_config = lstmConfig
        test_results = getPbDoResults(x=test_data, sim_y_pred= test_sim_label, y=test_label, phys_data=test_phys_data,
                  hypsography=hypsography, config=train_config, dataset='test', save_path=save_path)
        return test_results
    # # train model
    start_time = time.time()
    model.to(device)
    # model.before_train(trainConfig=train_config)
    model.before_train(trainConfig=train_config)
    set_seed(seed=seed)
    model.fit_stage3(x=trn_data, y=trn_label, unsup_x=test_data, phy_data=test_phys_data, val_x=val_data, val_y=val_label, hypsography = hypsography,
                                batch_size=train_config['batch_size'],
                                epochs=train_config['train_epochs'],config=train_config,
                                shuffle=False, use_gpu=True)

    ###### SAVE Results #####
    test_results = model.predict_do(x=test_data, y=test_label, phys_data=test_phys_data,
                  hypsography=hypsography, config=train_config, dataset='test', save_path=save_path)

    train_results = model.predict_do(x=trn_data, y=trn_label, phys_data=trn_phys_data,
                  hypsography=hypsography, config=train_config, dataset='train', save_path=save_path)

    val_results = model.predict_do(x=val_data, y=val_label, phys_data=val_phys_data,
                  hypsography=hypsography, config=train_config, dataset='val', save_path=save_path)

    torch.save(model.state_dict(), os.path.join(
        save_path, 'model' + '.pth')) 

    save_data(os.path.join(save_path, 'obs_train.npy'), trn_label)
    save_data(os.path.join(save_path, 'obs_test.npy'), test_label)
    save_data(os.path.join(save_path, 'obs_val.npy'), val_label)

    save_data(os.path.join(save_path, 'sim_train.npy'), trn_sim_label)
    save_data(os.path.join(save_path, 'sim_test.npy'), test_sim_label)
    save_data(os.path.join(save_path, 'sim_val.npy'), val_sim_label)

    save_data(os.path.join(save_path, 'train_phys.npy'), trn_phys_data)
    save_data(os.path.join(save_path, 'test_phys.npy'), test_phys_data)
    save_data(os.path.join(save_path, 'val_phys.npy'), val_phys_data)

    save_data(os.path.join(save_path, 'train_dates.npy'), trn_dates)
    save_data(os.path.join(save_path, 'test_dates.npy'), test_dates)
    save_data(os.path.join(save_path, 'val_dates.npy'), val_dates)

    end_time = time.time()
    cost_time = int(end_time - start_time)

    logging.info("Model Functioning period end")
    logging.info('Model Functioning period cost:' + str(cost_time))

    return test_results
