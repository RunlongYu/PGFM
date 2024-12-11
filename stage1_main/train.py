import argparse
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch PGFM(Foudation model) example')
    parser.add_argument('--model', type=str, default='FM',
                        help='use model', choices=['FM', 'FM_TF'])
    parser.add_argument('--stage1_label', type=str, default='sim', help='label of prediction column',
                        choices=['label', 'sim_hyp', 'sim_epi', 'sim_do', 'sim_temp', 'sim'])
    parser.add_argument('--dataset', type=str,
                        default='lake', help='use dataset')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--mutation', type=int, default=1,
                        help='use mutation: 1 use 0 not used')
    parser.add_argument('--strategy', type=str, default='n+1',
                        help='use strategy', choices=['1,1', '1+1', 'n,1', 'n+1'])
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')

    args = parser.parse_args()
    dataset = args.dataset
    if dataset == 'lake':
        from run.run_lake_fm import train
        
    train(params=args)
