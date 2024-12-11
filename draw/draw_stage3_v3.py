import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
import argparse
from utils.utils import set_seed
from stage3_scripts.finetune_do import load_data
from utils.utils import load_pd_data, set_seed
from utils.utils import combine_dataset
import numpy as np
import pandas as pd
from draw_utils import draw_do_new
from utils.utils import USE_FEATURES, UNSUP_TEMP_FEATURES

def draw_stage3(args):
    seed = args.seed
    model_type = args.model_type
    datetime = args.datetime
    label_name = 'obs_do'
    set_seed(seed)
    main_read_path = f'../stage3_results/{model_type}/{datetime}/'
    ids_pd = pd.read_csv(f'../utils/intersection_ids.csv')
    ids = ids_pd['nhdhr_id'].to_list()

    for lake_id in ids:
        read_path = os.path.join(main_read_path + f'{lake_id}/{seed}', 'pred_test.npy')
        pred_do = np.load(read_path)


        read_path_raw = os.path.join(main_read_path + f'{lake_id}/{seed}', 'pred_test_raw.npy')
        pred_do_raw = np.load(read_path_raw)

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
            stage2_data_test, stage2_raw_data_test, label_name,  lake_id, data_type='test', model_type = model_type)
        
        sim_temp = test_sim_label.squeeze()
        unique_depths = np.unique(stage2_raw_data_test['raw_vertical_depth'])
        n_depth = len(unique_depths)
        # n_depth = 2
        obs_do = test_label.squeeze()
        ice_flag_index = USE_FEATURES.index('ice')
        ice_flag = test_input[:,:,ice_flag_index].squeeze()
        pic_index = 0


        tcl_depth_index = USE_FEATURES.index('thermocline_depth')

        sim_do = test_sim_label.squeeze()
        obs_do = test_label.squeeze()
        
        tcl_depth = test_input[:,:,tcl_depth_index].squeeze()

        mix = np.zeros_like(tcl_depth)

        # 设置 stratified 的值，tcl_depth 为 0 的地方设为 0，其余地方设为 1
        mix[tcl_depth != 0] = 1
        print("test_input shape:", test_input.shape)
        print("pred_do shape:", pred_do.shape)
        print("pred_do_raw shape:", pred_do_raw.shape)

        pic_index = 0
        pred_index = 0
        for i in range(0, sim_do.shape[0], n_depth):
            epi = i
            hypo = i + n_depth
            print("hypo:", hypo)

            pred_epi = pic_index * 2
            pred_hypo = pred_epi + 1

            # pred_do_year = pred_do[[pred_index, pred_index+1], :].squeeze()
            # pred_index += 1

            # pred_do_year = pred_do_raw[[epi, hypo-1], :].squeeze()
            pred_do_year = pred_do[[pred_epi, pred_hypo], :].squeeze()

            date_year = test_dates[epi, :].squeeze()
            sim_do_year = sim_do[[epi, hypo-1], :]
            # sim_do_year = sim_do[epi: hypo, :]
            obs_year = obs_do[[epi, hypo-1], :]

            is_stratified = tcl_depth[epi, :].squeeze()
            is_stratified = np.where(is_stratified == 0, 0, 1)

            save_path = f'../pics/Oxygen_v3/{model_type}/{datetime}/{seed}/'
            os.makedirs(save_path, exist_ok=True)
            do_save_path = os.path.join(save_path, f'{model_type}_{lake_id}_seed{seed}_{pic_index}.pdf')
            draw_do_new(save_path=do_save_path, obs=obs_year, sim=sim_do_year, pred=pred_do_year, date=date_year, is_stratified=is_stratified, pic_model_name=model_type)
            pic_index += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process model type.")
    parser.add_argument("--model_type", type=str, default= 'fm-pg', help="get model type from input args", choices=['lstm', 'ealstm', 'transformer', 'fm-lstm', 'fm-transformer', 'fm-pg', 'fm-pg+'])
    parser.add_argument('--strategy', type=str, default='n+1', help='use strategy', choices=['1,1', '1+1', 'n,1', 'n+1'])
    # parser.add_argument("label_name", type=str, default= 'obs_temp', help="label name")
    parser.add_argument("--seed", type=int, default= 40, help="ramdom seed")
    parser.add_argument("--datetime", type=str, required=True, help="Start Time")
    args = parser.parse_args()
    draw_stage3(args)