import argparse
import sys
import os
from time import sleep
import pandas as pd
import time
import torch
########### MY PACKAGES ###########
sys.path.append(os.path.dirname(sys.path[0]))
from finetune import train
from utils.utils_stage2 import get_ids

def main(args):
        print("Seed:", args.seed)
        current_time = args.current_time
        ids = get_ids()

        result_csv_path = f'../stage2_results/{args.model_type}/{current_time}/{args.model_type}_{current_time}_seed{args.seed}_results.csv'
        results_df = pd.DataFrame(columns=['lake_id', 'sup_loss_all', 'sup_loss_winter', 'sup_loss_summer', 'dc_unsup_loss', 'ec_unsup_loss_all'])


        for lake_id in ids:
            args.lake_id = lake_id
            test_results = train(args)

            results = {
                'lake_id': lake_id,
                'sup_loss_all': test_results['sup_loss_all'].item() if torch.is_tensor(test_results['sup_loss_all']) else test_results['sup_loss_all'],
                'sup_loss_winter': test_results['sup_loss_winter'].item() if torch.is_tensor(test_results['sup_loss_winter']) else test_results['sup_loss_winter'],
                'sup_loss_summer': test_results['sup_loss_summer'].item() if torch.is_tensor(test_results['sup_loss_summer']) else test_results['sup_loss_summer'],
                'dc_unsup_loss': test_results['dc_unsup_loss'].item() if torch.is_tensor(test_results['dc_unsup_loss']) else test_results['dc_unsup_loss'],
                'ec_unsup_loss_all': test_results['ec_unsup_loss_all'].item() if torch.is_tensor(test_results['ec_unsup_loss_all']) else test_results['ec_unsup_loss_all']
            }
            results_df = pd.concat([results_df, pd.DataFrame([results])], ignore_index=True)
        os.makedirs(os.path.dirname(result_csv_path), exist_ok=True)

        results_df.to_csv(result_csv_path, index=False)
        print(f"Results saved to {result_csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process model type.")
    parser.add_argument("--model_type", type=str, default= 'fm-pg', help="get model type from input args", choices=['lstm', 'ealstm', 'transformer', 'fm-lstm', 'fm-transformer', 'fm-pg', 'PB'])
    parser.add_argument('--strategy', type=str, default='n+1', help='use strategy', choices=['1,1', '1+1', 'n,1', 'n+1'])
    parser.add_argument('--pt_datetime', type=str, default='2024-08-08-14-42', help='pretrain model datetime')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument("--label_name", type=str, default= 'obs_temp', help="label name")
    parser.add_argument("--seed", type=int, default= 40, help="ramdom seed")
    parser.add_argument("--current_time", type=str, required=True, help="Start Time")
    args = parser.parse_args()
    main(args)
