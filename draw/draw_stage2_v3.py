import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
import argparse
from utils.utils import set_seed
from stage2_scripts.finetune import load_data
from utils.utils import load_pd_data, make_data_temp, set_seed, make_data_phys
from utils.utils import combine_dataset
import numpy as np
import pandas as pd
from draw_utils import draw_temp_new
from utils.utils import USE_FEATURES, UNSUP_TEMP_FEATURES

def draw_stage2_v1(args):
    seed = args.seed
    model_type = args.model_type
    datetime = args.datetime
    draw_id = args.lake_id
    label_name = 'obs_temp'
    set_seed(seed)
    main_read_path = f'../stage2_results/{model_type}/{datetime}/'
    ids_pd = pd.read_csv(f'../utils/intersection_ids.csv')
    ids = ids_pd['nhdhr_id'].to_list()

    for lake_id in ids:
        if lake_id != draw_id:
            continue
        read_path = os.path.join(main_read_path + f'{lake_id}/{seed}', 'pred_test.npy')
        pred_temp = np.load(read_path)

        ids = [lake_id]
        test_pd_data = load_pd_data(
        ids=ids, dataset='test', save_path='../data/processedByLake/')
        stage1_data_test_do = test_pd_data['norm_do_data']
        stage1_data_test_raw_do = test_pd_data['raw_do_data']
        stage1_data_test_temp = test_pd_data['temp_data']
        print("-----Combine Test data------")
        stage2_data_test = combine_dataset(stage1_data_test_do, stage1_data_test_temp)
        stage2_raw_data_test = combine_dataset(stage1_data_test_raw_do, stage1_data_test_temp)

        test_input, test_label, test_sim_label, test_phys_data, _, test_dates = load_data(
            stage2_data_test, stage2_raw_data_test, label_name,  lake_id)
        
        sim_temp = test_sim_label.squeeze()
        unique_depths = np.unique(stage2_raw_data_test['raw_vertical_depth'])
        n_depth = len(unique_depths)
        obs_temp = test_label.squeeze()
        ice_flag_index = USE_FEATURES.index('ice')
        ice_flag = test_input[:,:,ice_flag_index].squeeze()
        pic_index = 0


        for i in range(0, pred_temp.shape[0], n_depth):
            if pic_index != 1:
                pic_index += 1
                continue
            start = i
            end = i + n_depth
            # depth = inputs[start:end, :, 0]
            depth = np.arange(0, n_depth * 0.5, 0.5)
            pred_year = pred_temp[start:end, :]
            date_year = test_dates[start, :]
            sim_year = sim_temp[start:end, :]
            obs_year = obs_temp[start:end, :]
            ice_flag_yaer = ice_flag[start:end, :]

            save_path = f'../pics/Temperature-v3/{model_type}/{datetime}/{seed}/'
            os.makedirs(save_path, exist_ok=True)
            temp_save_path = os.path.join(save_path, f'{model_type}_{lake_id}_seed{seed}_{pic_index}.pdf')




            draw_temp_new(temp_save_path, depth, obs_year, sim_year, pred_year, date_year, ice_flag=ice_flag_yaer, model_type=model_type)
            pic_index += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process model type.")
    parser.add_argument("--model_type", type=str, default= 'fm-pg', help="get model type from input args", choices=['lstm', 'ealstm', 'transformer', 'fm-lstm', 'fm-transformer', 'fm-pg'])
    parser.add_argument('--strategy', type=str, default='n+1', help='use strategy', choices=['1,1', '1+1', 'n,1', 'n+1'])
    # parser.add_argument("label_name", type=str, default= 'obs_temp', help="label name")
    parser.add_argument("--seed", type=int, default= 40, help="ramdom seed")
    parser.add_argument("--datetime", type=str, required=True, help="Start Time")
    parser.add_argument("--lake_id", type=str, required=True, help="Start Time")
    args = parser.parse_args()
    draw_stage2_v1(args)


