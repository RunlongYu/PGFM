import argparse
import sys
import os
from time import sleep
import pandas as pd
import time
import torch
########### MY PACKAGES ###########
sys.path.append(os.path.dirname(sys.path[0]))
from finetune_do import train
from tqdm import tqdm

def main(args):
        print("Seed:", args.seed)
        # current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        current_time = args.current_time
        ids_pd = pd.read_csv(f'../utils/intersection_ids.csv')
        ids = ids_pd['nhdhr_id'].to_list()

        result_csv_path = f'../stage3_results/{args.model_type}/{current_time}/{args.model_type}_{current_time}_seed{args.seed}_do_results.csv'
        results_df = pd.DataFrame(columns=['lake_id', 'sup_loss_all', 'sup_loss_mixed', 'sup_loss_epi', 'sup_loss_hypo', 'total_DO_loss','upper_DO_loss','lower_DO_loss'])

        for lake_id in tqdm(ids):
            args.lake_id = lake_id
            args.current_time = current_time
            test_results = train(args)
            results = {
                'lake_id': lake_id,
                'sup_loss_all': test_results['sup_loss_all'].item() if torch.is_tensor(test_results['sup_loss_all']) else test_results['sup_loss_all'],
                'sup_loss_mixed': test_results['sup_loss_mixed'].item() if torch.is_tensor(test_results['sup_loss_mixed']) else test_results['sup_loss_mixed'],
                'sup_loss_epi': test_results['sup_loss_epi'].item() if torch.is_tensor(test_results['sup_loss_epi']) else test_results['sup_loss_epi'],
                'sup_loss_hypo': test_results['sup_loss_hypo'].item() if torch.is_tensor(test_results['sup_loss_hypo']) else test_results['sup_loss_hypo'],
                'total_DO_loss': test_results['total_DO_loss'].item() if torch.is_tensor(test_results['total_DO_loss']) else test_results['total_DO_loss'],
                'upper_DO_loss': test_results['upper_DO_loss'].item() if torch.is_tensor(test_results['upper_DO_loss']) else test_results['upper_DO_loss'],
                'lower_DO_loss': test_results['lower_DO_loss'].item() if torch.is_tensor(test_results['lower_DO_loss']) else test_results['lower_DO_loss'],
            }
            results_df = pd.concat([results_df, pd.DataFrame([results])], ignore_index=True)

        os.makedirs(os.path.dirname(result_csv_path), exist_ok=True)
        results_df.to_csv(result_csv_path, index=False)
        print(f"Results saved to {result_csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process model type.")
    parser.add_argument("--model_type", type=str, default= 'fm-pg', help="get model type from input args", choices=['lstm', 'ealstm', 'transformer', 'fm-lstm', 'fm-ealstm', 'fm-transformer', 'fm-pg', 'fm-pg+', 'PB'])
    parser.add_argument('--strategy', type=str, default='n+1', help='use strategy', choices=['1,1', '1+1', 'n,1', 'n+1'])
    parser.add_argument('--pt_datetime', type=str, default='2024-08-08-14-42', help='pretrain model datetime')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument("--label_name", type=str, default= 'obs_do', help="label name")
    parser.add_argument("--seed", type=int, default= 40, help="ramdom seed")
    parser.add_argument("--current_time", type=str, required=True, help="Training time")
    args = parser.parse_args()
    main(args)
