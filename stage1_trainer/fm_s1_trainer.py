import os
import time
import sys
import logging
import pickle as pkl
import torch
import pandas as pd
from tqdm import tqdm
from utils.function_utils import get_feature_names
from config.configs import FM_Config, General_Config
from utils.function_utils import get_param_sum
from model.FM_Model import FM_Model
from model.FM_TF import FM_TF
import numpy as np
from utils.utils import set_seed, make_data
from utils.function_utils import random_selected_interaction_type
from utils.function_utils import seqLength
from sklearn.metrics import mean_squared_error, roc_auc_score, mean_absolute_error, mean_absolute_percentage_error
from datetime import datetime
import gc

# Get current time
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
# Format filenames with the current time
filename_predict = f"predict_result_{current_time}_s1.npy"
filename_label = f"label_result_{current_time}_s1.npy"
filename_obs = f"obs_result_{current_time}_s1.npy"


def evolution_search(feature_columns, data_train, data_test, param_save_dir, label_name, obs, embedding_size,
                     mutation=True, device='cpu', strategy='1+1', data_val=None, seed=None, model_type=None):
    set_seed(seed)
    logging.info('Cognitive EvoLutionary Search period param:')
    logging.info(FM_Config['FM'])

    feature_names = feature_columns

    # Random initialize interaction_type
    pair_feature_len = int(len(feature_names) * (len(feature_names)-1) / 2)
    selected_interaction_type = random_selected_interaction_type(
        pair_feature_len)

    # Make training data and validation data
    print("Processing model input and label for traninig set...")
    train_input, train_label, _ = make_data(data_train, label_name, feature_names)
    # val_input, val_label, _ = make_data(data_val, label_name, feature_names)
    # test_input, test_label, _ = make_data(data_test, label_name, feature_names)
    val_input = None
    val_label = None

    logging.info('Cognitive EvoLutionary Search period start')
    logging.info(f'Model type: {model_type}')
    if model_type == 'FM':
        FM_model = FM_Model(feature_columns=feature_columns,
                                param_save_dir=param_save_dir,
                                selected_interaction_type=selected_interaction_type,
                                mutation=mutation,
                                mutation_probability=FM_Config['FM']['mutation_probability'],
                                embedding_size=embedding_size, seed=seed,
                                device=device)
    elif model_type == 'FM_TF':
        FM_model = FM_TF(feature_columns=feature_columns,
                                param_save_dir=param_save_dir,
                                selected_interaction_type=selected_interaction_type,
                                mutation=mutation,
                                mutation_probability=FM_Config['FM']['mutation_probability'],
                                embedding_size=embedding_size, seed=seed,
                                device=device)

    FM_model.to(device)
    FM_model.before_train()
    start_time = time.time()
    get_param_sum(model=FM_model)

    if strategy == '1,1':
        FM_model.fit_1_1(train_input, train_label,
                           batch_size=General_Config['general']['batch_size'],
                           epochs=General_Config['general']['epochs'],
                           validation_split=(val_input, val_label),
                           shuffle=False)
    elif strategy == '1+1':
        FM_model.fit_1_plus_1(train_input, train_label,
                                batch_size=General_Config['general']['batch_size'],
                                epochs=General_Config['general']['epochs'],
                                validation_split=(val_input, val_label),
                                shuffle=False)

    elif strategy == 'n,1':
        FM_model.fit_n_1(train_input, train_label,
                           batch_size=General_Config['general']['batch_size'],
                           epochs=General_Config['general']['epochs'],
                           validation_split=(val_input, val_label),
                           shuffle=False)
    elif strategy == 'n+1':
        FM_model.fit_n_plus_1(train_input, train_label, val_x=val_input, val_y=val_label,
                                batch_size=General_Config['general']['batch_size'],
                                epochs=General_Config['general']['epochs'],
                                shuffle=False)

    FM_model.after_train(param_save_dir=param_save_dir)

    torch.save(FM_model.state_dict(), os.path.join(
        param_save_dir, 'model' + '.pth'))