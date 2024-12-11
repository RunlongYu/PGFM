import os
import torch
import pandas as pd
import numpy as np
from utils.function_utils import seqLength
from config.configs import CELS_Config, General_Config
from tqdm import tqdm
from torch.utils.data import DataLoader
from config.temp_config import fmPgConfig, fmLSTMConfig
from scipy import interpolate
import torch.nn as nn
import torch.utils.data as Data
import random
import logging
import sys
NORM_COLUMNS = ['sat_hypo', 'thermocline_depth', 'temperature_epi', 'temperature_hypo', 'volume_epi', 'volume_hypo', 'fnep',
                'fmineral', 'fsed', 'fatm', 'fdiff', 'fentr_epi', 'fentr_hyp',
                'eutro', 'oligo', 'dys', 'water', 'developed', 'barren', 'forest',
                'shrubland', 'herbaceous', 'cultivated', 'wetlands', 'depth', 'area', 'elev',
                'Shore_len', 'Vol_total', 'Vol_res', 'Vol_src', 'Depth_avg', 'Dis_avg', 'Res_time',
                'Elevation', 'Slope_100', 'Wshd_area', 'ShortWave', 'LongWave', 'AirTemp', 'RelHum',
                'WindSpeed', 'Rain', 'Snow']


USE_FEATURES = ['sat_hypo', 'thermocline_depth', 'temperature_epi', 'temperature_hypo', 'volume_epi', 'volume_hypo', 'fnep',
                'fmineral', 'fsed', 'fatm', 'fdiff', 'fentr_epi', 'fentr_hyp',
                'eutro', 'oligo', 'dys', 'water', 'developed', 'barren', 'forest',
                'shrubland', 'herbaceous', 'cultivated', 'wetlands', 'depth', 'area', 'elev',
                'Shore_len', 'Vol_total', 'Vol_res', 'Vol_src', 'Depth_avg', 'Dis_avg', 'Res_time',
                'Elevation', 'Slope_100', 'Wshd_area', 'ice', 'ShortWave', 'LongWave', 'AirTemp', 'RelHum',
                'WindSpeed', 'Rain', 'Snow', 'vertical_depth', 'layer']

tcl_depth_index = USE_FEATURES.index('thermocline_depth')


UNSUP_TEMP_FEATURES = ['vertical_depth', 'ShortWave', 'LongWave',
                       'AirTemp', 'RelHum', 'WindSpeed', 'Rain', 'Snow', 'ice']


UNSUP_DO_FEATURES = ['volume_epi','volume_hypo','fnep', 'fmineral', 'fsed', 'fatm', 'fdiff', 'fentr_epi', 'fentr_hyp']


DROP_COLUMNS = ['AF', 'geometry', 'Hylak_id', 'Grand_id', 'Lake_name', 'Country', 'Continent',
                'Hylak_id', 'Lake_type', 'Grand_id', 'n_epi', 'n_hypo', 'n_mixed', 'fit_train', 'fit_test',
                'fit_all', 'obs_total', 'mean_prob_dys', 'var_prob_dys', 'mean_prob_eumixo',
                'var_prob_eumixo', 'mean_prob_oligo', 'var_prob_oligo', 'Poly_src',
                'NEP_mgm3d', 'SED_mgm2d', 'MIN_mgm3d', 'khalf', 'Pour_long', 'Pour_lat', 'Lake_area', 'Shore_dev',
                'ct', 'categorical_ts']

# For ea-lstm
Dynamic_FEATURES = ['sat_hypo', 'thermocline_depth', 'temperature_epi', 'temperature_hypo', 'volume_epi', 'volume_hypo', 'fnep',
                'fmineral', 'fsed', 'fatm', 'fdiff', 'fentr_epi', 'fentr_hyp',
                'eutro', 'oligo', 'dys', 'water', 'developed', 'barren', 'forest',
                'shrubland', 'herbaceous', 'cultivated', 'wetlands', 'depth', 'area', 'elev',
                'Shore_len', 'Vol_total', 'Vol_res', 'Vol_src', 'Depth_avg', 'Dis_avg', 'Res_time',
                'Elevation', 'Slope_100', 'Wshd_area', 'ice', 'ShortWave', 'LongWave', 'AirTemp', 'RelHum',
                'WindSpeed', 'Rain', 'Snow', 'vertical_depth', 'layer']

STATIC_FEATURES = ['depth', 'area', 'elev','Shore_len', 'Vol_total', 'Vol_res', 'Vol_src','Depth_avg', 'Dis_avg', 'Res_time', 'Elevation', 'Slope_100', 'Wshd_area']

start_date_list = {'train': pd.to_datetime(
    '1980-01-01'), 'val': pd.to_datetime('2012-01-01'), 'test': pd.to_datetime('2016-01-01')}
end_date_list = {'train': pd.to_datetime(
    '2011-12-31'), 'val': pd.to_datetime('2015-12-31'), 'test': pd.to_datetime('2019-12-31')}

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    print(f"Random seed set as {seed}")

def load_pd_data(ids, dataset, save_path):
    data = {'raw_do_data': {}, 'norm_do_data': {}, 'temp_data': {}}
    all_raw_do_data = pd.DataFrame()
    all_norm_do_data = pd.DataFrame()
    all_temp_data = pd.DataFrame()
    print(f"Reading data in {dataset} dataset...")
    for lake_id in tqdm(ids):
        save_path_lake = save_path + lake_id
        raw_do_data = pd.read_csv(os.path.join(
            save_path_lake, f'{dataset}_raw_do_data.csv'))
        norm_do_data = pd.read_csv(os.path.join(
            save_path_lake, f'{dataset}_norm_do_data.csv'))
        temp_data = pd.read_csv(os.path.join(
            save_path_lake, f'{dataset}_temp_data.csv'))

        norm_do_data['raw_thermocline_depth'] = raw_do_data['thermocline_depth']
        raw_do_data['raw_thermocline_depth'] = raw_do_data['thermocline_depth']
        
        all_raw_do_data = pd.concat(
            [all_raw_do_data, raw_do_data], ignore_index=True)
        all_norm_do_data = pd.concat(
            [all_norm_do_data, norm_do_data], ignore_index=True)
        all_temp_data = pd.concat(
            [all_temp_data, temp_data], ignore_index=True)

    data['raw_do_data'] = all_raw_do_data
    data['norm_do_data'] = all_norm_do_data
    data['temp_data'] = all_temp_data
    return data

def combine_dataset(do_data, temp_data):
    combined_data = []
    grouped_do_data = do_data.groupby('nhdhr_id')
    grouped_temp_data = temp_data.groupby('nhdhr_id')

    for lake_id, sub_do_data in tqdm(grouped_do_data):
        if lake_id in grouped_temp_data.groups:
            sub_temp_data = grouped_temp_data.get_group(lake_id)
        else:
            continue

        combined_sub_data = extend_combine(sub_do_data, sub_temp_data)

        combined_data.append(combined_sub_data)


    if combined_data:
        combined_data = pd.concat(combined_data, ignore_index=True)
    else:
        combined_data = pd.DataFrame()

    return combined_data

def extend_combine(do_data, temp_data):
    # print("--------------------------")

    unique_depths = np.unique(temp_data['raw_vertical_depth'])
    n_depth = len(unique_depths)
    # print("unique_depths:", unique_depths)
    # print("n_depth:", n_depth)


    extended_do_data = do_data.loc[np.repeat(
        do_data.index, n_depth)].reset_index(drop=True)
    temp_data = temp_data.drop(['datetime', 'nhdhr_id'], axis=1)



    temp_data.reset_index(drop=True, inplace=True)
    extended_do_data.reset_index(drop=True, inplace=True)

    combined_data = pd.concat([temp_data, extended_do_data], axis=1)
    combined_data['layer'] = np.where(
        combined_data['raw_vertical_depth'] < combined_data['raw_thermocline_depth'], 0, 1)

    combined_data.loc[combined_data['raw_thermocline_depth'].isna(), 'layer'] = 0
    combined_data.loc[combined_data['raw_thermocline_depth'] == 0, 'layer'] = 0
    combined_data['sim_do'] = np.where(
        combined_data['layer'] == 0,
        combined_data['sim_epi'],
        combined_data['sim_hyp']
    )
    combined_data['obs_do'] = np.where(
        combined_data['layer'] == 0,
        combined_data['obs_epi'],
        combined_data['obs_hyp']
    )
    # print(combined_data.columns)
    # print("length:", len(combined_data.columns))
    return combined_data

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def _GetBatch(Data, dates, seq_length, win_shift):
    # Data = np.concatenate((inputs, labels), axis=2)
    n_layer, n_dates, n_features = Data.shape
    # seq_per_layer = math.floor(n_dates / seq_length)

    seq_per_layer = (n_dates - seq_length) // win_shift + \
        1    # numer of batches
    n_seq = seq_per_layer * n_layer
    Data_batch = np.empty(
        shape=(n_seq, seq_length, n_features))  # features + label
    Dates_batch = np.empty(shape=(n_seq, seq_length), dtype='datetime64[s]')
    Data_batch[:] = np.nan
    Dates_batch[:] = np.datetime64("NaT")
    ct = 0
    for seq in range(seq_per_layer):
        for l in range(n_layer):
            start_idx = seq * win_shift
            end_idx = start_idx + seq_length
            Data_batch[ct, :, :] = Data[l, start_idx:end_idx, :]
            Dates_batch[ct, :] = dates[l, start_idx:end_idx]
            ct += 1

    valid_indices = []
    for seq in range(0, n_seq, n_layer):  # step by n_layer to check groups
        target_group = Data_batch[seq: seq + n_layer, :, -1]
        if np.any(~np.isnan(target_group)):  # Check if there is at least one non-NaN in the group
            valid_indices.extend(range(seq, seq + n_layer))

    Data_batch = Data_batch[valid_indices, :, :]
    Dates_batch = Dates_batch[valid_indices, :]

    for i in range(Data_batch.shape[0]):
        assert not np.isnan(Data_batch[i]).all(
        ), f"Row {i} in Data_batch is all NaN"

    return torch.from_numpy(Data_batch), Dates_batch

def make_data(raw_data, label_names, feature_names):
    seq_length = seqLength  # 365 days
    data_grouped = raw_data.groupby('nhdhr_id')
    if label_names == 'sim':
        label_names = ['sim_do', 'sim_temp']
        print("label_names:", label_names)
    elif label_names == 'all':
        label_names = ['sim_do', 'sim_temp', 'obs_do', 'obs_temp']
        print("label_names:", label_names)
    else:
        label_names = [label_names]

    all_input = []
    all_label = []
    all_dates = []
    for lake_id, data_pd in tqdm(data_grouped):
        unique_depths = np.unique(data_pd['raw_vertical_depth'])
        n_depth = len(unique_depths)
        # print("norm depth:")
        data_pd['vertical_depth'] = data_pd['vertical_depth'] + CELS_Config['CELS']['depth_shift']
        data_pd['vertical_depth'] = data_pd['vertical_depth']/CELS_Config['CELS']['depth_zoom']
        # data_pd['vertical_depth'] = data_pd['vertical_depth'].replace(0, 1e-6)
        
        # data_pd['layer'] = data_pd['layer'].replace(0, 1e-6)
        # data_pd['layer'] = data_pd['layer'].replace(1, 1e-1)
        data_pd['layer'] = 0

        data = data_pd[feature_names + label_names].to_numpy()
        dates = data_pd['datetime'].to_numpy()

        n_samples = data.shape[0] // n_depth
        new_shape = (n_samples, n_depth, data.shape[1])
        data = data.reshape(new_shape).transpose(1, 0, 2)
        dates = dates.reshape(n_samples, n_depth).transpose(1, 0)
        data_batch, date_batch = _GetBatch(
            data, dates, seq_length, win_shift=seq_length)
        

        input = data_batch[:, :, :len(feature_names)]
        label = data_batch[:, :, -len(label_names):]
        # print("unique_depths:", unique_depths)
        # print("depth:", input[:n_depth,0,-1])
        all_input.append(input)
        all_label.append(label)
        all_dates.append(date_batch)

    all_input = np.concatenate(all_input, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_dates = np.concatenate(all_dates, axis=0)

    print("----- Shape -----")
    print("input shape:", all_input.shape)
    print("label shape:", all_label.shape)
    print("dates shape:", all_dates.shape)

    # all_input = all_input.reshape(-1, all_input.shape[-1])
    # all_label = all_label.reshape(-1, all_label.shape[-1])
    # print("-----After reshape-----")
    # print("all_input shape:", all_input.shape)
    # print("all_label shape:", all_label.shape)
    # all_input_df = pd.DataFrame(all_input, columns=feature_names)

    return all_input, all_label, all_dates

def make_data_temp(raw_data, label_names, feature_names):
    seq_length = seqLength  # 365 days
    data_grouped = raw_data.groupby('nhdhr_id')

    if label_names == 'obs_do':
        label_names = ['sim_do','obs_do']
    else:
        label_names = ['sim_temp', 'obs_temp']

    print("Label name:", label_names)
    all_input = []
    all_label = []
    all_dates = []
    for lake_id, data_pd in tqdm(data_grouped):
        unique_depths = np.unique(data_pd['raw_vertical_depth'])
        n_depth = len(unique_depths)
        # print("norm depth:")
        data_pd['vertical_depth'] = data_pd['vertical_depth'] + CELS_Config['CELS']['depth_shift']
        data_pd['vertical_depth'] = data_pd['vertical_depth']/CELS_Config['CELS']['depth_zoom']
        # data_pd['vertical_depth'] = data_pd['vertical_depth'].replace(0, 1e-6)
        
        # data_pd['layer'] = data_pd['layer'].replace(0, 1e-6)
        # data_pd['layer'] = data_pd['layer'].replace(1, 1e-1)
        data_pd['layer'] = 0

        data = data_pd[feature_names + label_names].to_numpy()
        dates = data_pd['datetime'].to_numpy()

        n_samples = data.shape[0] // n_depth
        new_shape = (n_samples, n_depth, data.shape[1])
        data = data.reshape(new_shape).transpose(1, 0, 2)
        dates = dates.reshape(n_samples, n_depth).transpose(1, 0)
        data_batch, date_batch = _GetBatch(
            data, dates, seq_length, win_shift=seq_length)
        
        
        input = data_batch[:, :, :len(feature_names)]
        label = data_batch[:, :, -len(label_names):]
        # print("unique_depths:", unique_depths)
        # print("depth:", input[:n_depth,0,-1])
        all_input.append(input)
        all_label.append(label)
        all_dates.append(date_batch)

    all_input = np.concatenate(all_input, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_dates = np.concatenate(all_dates, axis=0)

    print("----- Shape -----")
    print("input shape:", all_input.shape)
    print("label shape:", all_label.shape)
    print("dates shape:", all_dates.shape)

    # all_input = all_input.reshape(-1, all_input.shape[-1])
    # all_label = all_label.reshape(-1, all_label.shape[-1])
    # print("-----After reshape-----")
    # print("all_input shape:", all_input.shape)
    # print("all_label shape:", all_label.shape)
    # all_input_df = pd.DataFrame(all_input, columns=feature_names)

    return all_input, all_label, all_dates

def calculate_weighted_avg_temp(input, temp_pred, feature_names, depth_areas, n_depth):
    layer_index = feature_names.index('layer')
    layer_info = input[:, :, layer_index]

    temp_epi_list = []
    temp_hypo_list = []

    for i in range(0, input.shape[0], n_depth):
        start = i
        end = i + n_depth

        temp_year = temp_pred[start:end, :]
        layer_info_year = layer_info[start:end, :]

        temp_epi = np.zeros((n_depth, temp_year.shape[1]))
        temp_hypo = np.zeros((n_depth, temp_year.shape[1]))

        for day in range(temp_year.shape[1]):
            temp_day = temp_year[:, day]
            layer_day = layer_info_year[:, day]

            # Calculate the weighted average for the upper layer
            upper_mask = (layer_day == 1)
            upper_temps = temp_day[upper_mask]
            upper_areas = depth_areas[upper_mask]
            if upper_temps.size > 0:
                temp_epi_value = np.average(upper_temps, weights=upper_areas)
                temp_epi[:, day] = temp_epi_value
            else:
                temp_epi[:, day] = 0

            # Calculate the weighted average for the lower layer
            lower_mask = (layer_day == 0)
            lower_temps = temp_day[lower_mask]
            lower_areas = depth_areas[lower_mask]
            if lower_temps.size > 0:
                temp_hypo_value = np.average(lower_temps, weights=lower_areas)
                temp_hypo[:, day] = temp_hypo_value
            else:
                temp_hypo[:, day] = 0

        temp_epi_list.append(temp_epi)
        temp_hypo_list.append(temp_hypo)

    # Concatenate all years' data into a single array
    temp_epi_all_years = np.concatenate(temp_epi_list, axis=0)
    temp_hypo_all_years = np.concatenate(temp_hypo_list, axis=0)

    # Standardize the results using the mean and standard deviation
    mean_dic = np.load('../data/processedByLake/mean_feats.npy', allow_pickle=True).item()
    std_dic = np.load('../data/processedByLake/std_feats.npy', allow_pickle=True).item()

    temp_epi_mean = mean_dic['temperature_epi']
    temp_epi_std = std_dic['temperature_epi']

    temp_hypo_mean = mean_dic['temperature_hypo']
    temp_hypo_std = std_dic['temperature_hypo']

    temp_epi_all_years = (temp_epi_all_years - temp_epi_mean) / temp_epi_std
    temp_hypo_all_years = (temp_hypo_all_years - temp_hypo_mean) / temp_hypo_std

    return temp_epi_all_years, temp_hypo_all_years

def make_data_do(raw_data, label_names, feature_names, depth_areas, model_type, data_type):
    seq_length = seqLength  # 365 days
    print("label_names:", label_names)
    data_grouped = raw_data.groupby('nhdhr_id')
    
    label_names = ['sim_do','obs_do']

    all_input = []
    all_label = []
    all_dates = []
    for lake_id, data_pd in tqdm(data_grouped):
        unique_depths = np.unique(data_pd['raw_vertical_depth'])
        n_depth = len(unique_depths)
        # print("norm depth:")
        data_pd['vertical_depth'] = data_pd['vertical_depth'] + CELS_Config['CELS']['depth_shift']
        data_pd['vertical_depth'] = data_pd['vertical_depth']/CELS_Config['CELS']['depth_zoom']

        data = data_pd[feature_names + label_names].to_numpy()
        dates = data_pd['datetime'].to_numpy()

        n_samples = data.shape[0] // n_depth
        new_shape = (n_samples, n_depth, data.shape[1])
        data = data.reshape(new_shape).transpose(1, 0, 2)
        dates = dates.reshape(n_samples, n_depth).transpose(1, 0)
        data_batch, date_batch = _GetBatch(
            data, dates, seq_length, win_shift=seq_length)

        input = data_batch[:, :, :len(feature_names)]
        label = data_batch[:, :, -len(label_names):]
        if model_type == 'fm-pg+':
            if input.shape[0] != 0:
                temp_epi_index = feature_names.index('temperature_epi')
                temp_hypo_index = feature_names.index('temperature_hypo')
                temp_pred = np.load(f'../data/temp_by_model/{lake_id}/pred_{data_type}.npy')
                temp_epi_avg, temp_hypo_avg = calculate_weighted_avg_temp(input, temp_pred, feature_names, depth_areas, n_depth)
                input[:,:,temp_epi_index] = torch.from_numpy(temp_epi_avg)
                input[:,:,temp_hypo_index] = torch.from_numpy(temp_hypo_avg)
        
        all_input.append(input)
        all_label.append(label)
        all_dates.append(date_batch)

    all_input = np.concatenate(all_input, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_dates = np.concatenate(all_dates, axis=0)

    print("----- Shape -----")
    print("input shape:", all_input.shape)
    print("label shape:", all_label.shape)
    print("dates shape:", all_dates.shape)

    # all_input = all_input.reshape(-1, all_input.shape[-1])
    # all_label = all_label.reshape(-1, all_label.shape[-1])
    # print("-----After reshape-----")
    # print("all_input shape:", all_input.shape)
    # print("all_label shape:", all_label.shape)
    # all_input_df = pd.DataFrame(all_input, columns=feature_names)

    return all_input, all_label, all_dates

def make_data_phys(raw_data, label_names, feature_names):
    seq_length = seqLength  # 365 days
    data_grouped = raw_data.groupby('nhdhr_id')
    if label_names == 'sim':
        label_names = ['sim_do', 'sim_temp']
        print("label_names:", label_names)
    elif label_names == 'all':
        label_names = ['sim_do', 'sim_temp', 'obs_do', 'obs_temp']
        print("label_names:", label_names)
    else:
        label_names = [label_names]

    all_input = []
    all_label = []
    all_dates = []
    for lake_id, data_pd in tqdm(data_grouped):
        unique_depths = np.unique(data_pd['raw_vertical_depth'])
        n_depth = len(unique_depths)

        data_pd['layer'] = 0

        data = data_pd[feature_names + label_names].to_numpy()
        dates = data_pd['datetime'].to_numpy()

        n_samples = data.shape[0] // n_depth
        new_shape = (n_samples, n_depth, data.shape[1])
        data = data.reshape(new_shape).transpose(1, 0, 2)
        dates = dates.reshape(n_samples, n_depth).transpose(1, 0)
        data_batch, date_batch = _GetBatch(
            data, dates, seq_length, win_shift=seq_length)
        input = data_batch[:, :, :len(feature_names)]
        label = data_batch[:, :, -len(label_names):]
        # print("unique_depths:", unique_depths)
        # print("depth:", input[:n_depth,0,-1])
        all_input.append(input)
        all_label.append(label)
        all_dates.append(date_batch)

    all_input = np.concatenate(all_input, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_dates = np.concatenate(all_dates, axis=0)

    print("----- Shape -----")
    print("input shape:", all_input.shape)
    print("label shape:", all_label.shape)
    print("dates shape:", all_dates.shape)

    # all_input = all_input.reshape(-1, all_input.shape[-1])
    # all_label = all_label.reshape(-1, all_label.shape[-1])
    # print("-----After reshape-----")
    # print("all_input shape:", all_input.shape)
    # print("all_label shape:", all_label.shape)
    # all_input_df = pd.DataFrame(all_input, columns=feature_names)

    return all_input, all_label, all_dates

def getHypsographyManyLakes(path, lakename, depths):
    # my_path = os.path.abspath(os.path.dirname(__file__))
    # full_path = os.path.join(my_path, path)
    if not os.path.exists(path):
        print("full_path:", path)
        print("no hypsography file")
        raise Exception()
        return None
    df = pd.read_csv(path, header=0, index_col=0)

    if df.shape[1] == 1:
        df = df.iloc[:, 0]

    depth_areas = df.to_dict()

    # print("depth_areas:", depth_areas)
    if len(depth_areas) < 3:
        # new try
        avail_depths = np.array(list(depth_areas.keys()))
        avail_areas = np.array(list(depth_areas.values()))
        sort_ind = np.argsort(avail_depths)
        avail_depths = avail_depths[sort_ind]
        avail_areas = avail_areas[sort_ind]

        f = interpolate.interp1d(
            avail_depths, avail_areas,  fill_value="extrapolate")
        new_depth_areas = np.array([f(x) for x in depths])
        # print("new_depth_areas:", new_depth_areas)
        # print("new_depth_areas shape:", new_depth_areas.shape)
        return new_depth_areas

    else:
        # old try
        tmp = {}
        total_area = 0
        for key, val in depth_areas.items():
            total_area += val

        for depth in depths:
            # find depth with area that is closest
            depth_w_area = min(list(depth_areas.keys()),
                               key=lambda x: abs(x-depth))
            tmp[depth] = depth_areas[depth_w_area]
        depth_areas = {}

        for k, v in tmp.items():
            total_area += v

        for k, v in tmp.items():
            depth_areas[k] = tmp[k]

        return np.sort(-np.array([list(depth_areas.values())]))*-1

def get_combined_do(y_pred, layer):
    """
    Computes the combined output for the upper and lower layers based on the predictions and layer information.

    Parameters:
    y_pred (numpy.ndarray): A 2D array of shape (num_layers, num_time_steps), representing the predictions for each layer across different time steps.
    layer (numpy.ndarray): A 2D array of shape (num_layers, num_time_steps), representing the layer types (0 for upper layer, 1 for lower layer).

    Returns:
    numpy.ndarray: A 2D array of shape (2, num_time_steps), where the first row represents the combined result for the upper layer,
                   and the second row represents the combined result for the lower layer.
    """
    # Get the shape of the input arrays
    num_layers, num_time_steps = y_pred.shape
    
    # Initialize the output array; the first row will be for the upper layer and the second row for the lower layer
    combined_do = np.zeros((2, num_time_steps))  # 2 represents the upper and lower layers

    for t in range(num_time_steps):
        # Get the current layer and prediction data at time step t
        current_layer = layer[:, t]
        current_pred = y_pred[:, t]

        # Create masks to separate upper and lower layer data
        upper_layer_mask = (current_layer == 0)
        lower_layer_mask = (current_layer == 1)

        # Extract the prediction values for the upper and lower layers
        upper_layer_data = current_pred[upper_layer_mask]
        lower_layer_data = current_pred[lower_layer_mask]

        # Calculate the combined output for the upper layer
        if len(upper_layer_data) > 0:
            # If there is more than one value, use a weighted sum; otherwise, take the single value directly
            combined_do[0, t] = 0.9 * upper_layer_data[0] + 0.1 * np.sum(upper_layer_data[1:]) / (len(upper_layer_data) - 1) if len(upper_layer_data) > 1 else upper_layer_data[0]
        
        # By default, copy the upper layer's result to the lower layer as a baseline
        combined_do[1, t] = combined_do[0, t]

        # Calculate the combined output for the lower layer (if there is data)
        if len(lower_layer_data) > 0:
            # Similar logic as above, but for the lower layer
            combined_do[1, t] = 0.9 * lower_layer_data[-1] + 0.1 * np.sum(lower_layer_data[:-1]) / (len(lower_layer_data) - 1) if len(lower_layer_data) > 1 else lower_layer_data[-1]

    return combined_do