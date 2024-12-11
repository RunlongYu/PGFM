import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from itertools import combinations
from collections import OrderedDict

seqLength = 365

def calculate_l1_loss(model):
    def l1_loss(x):
        return torch.abs(x).sum()

    to_regularize = []
    for name, p in model.named_parameters():
        if 'bias' in name:
            continue
        else:
            to_regularize.append(p.view(-1))
    l1_loss_val = l1_loss(torch.cat(to_regularize))
    return l1_loss_val


def generate_pair_index(n, order=2, selected_pairs=None):
    """Return enumeration of feature combination pair index.

    :param n: number of valid features, usually equals to `input_dim4lookup`
    :type n: int
    :param order: order of interaction. defaults to 2
    :type order: int
    :param selected_pairs: specifying selected pair of index
    :type selected_pairs: sequence of tuples, optional
    :return: a list of tuple, each containing feature index
    :rtype: list of tuple

    :Example:

    >>> generate_pair_index(5, 2)
    >>> [(0, 0, 0, 0, 1, 1, 1, 2, 2, 3),
         (1, 2, 3, 4, 2, 3, 4, 3, 4, 4)]
    >>> generate_pair_index(5, 3)
    >>> [(0, 0, 0, 0, 0, 0, 1, 1, 1, 2),
         (1, 1, 1, 2, 2, 3, 2, 2, 3, 3),
         (2, 3, 4, 3, 4, 4, 3, 4, 4, 4)]
    >>> generate_pair_index(5, 2, [(0,1),(1,3),(2,3)])
    >>> [(0, 1, 2), (1, 3, 3)]
    """
    if n < 2:
        raise ValueError("undefined. please ensure n >= 2")
    pairs = list(combinations(range(n), order))
    if selected_pairs is not None and len(selected_pairs) > 0:
        valid_pairs = set(selected_pairs)
        pairs = list(filter(lambda x: x in valid_pairs, pairs))
        print("Using following selected feature pairs \n{}".format(pairs))
        if len(pairs) != len(selected_pairs):
            print("Pair number {} != specified pair number {}".format(len(pairs), len(selected_pairs)))
    return list(zip(*pairs))


def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return torch.cat(inputs, dim=axis)


def combined_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = torch.flatten(
            torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
        dense_dnn_input = torch.flatten(
            torch.cat(dense_value_list, dim=-1), start_dim=1)
        return concat_fun([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
    elif len(dense_value_list) > 0:
        return torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
    else:
        raise NotImplementedError


def slice_arrays(arrays, start=None, stop=None):
    if arrays is None:
        return [None]

    if isinstance(arrays, np.ndarray):
        arrays = [arrays]

    if isinstance(start, list) and stop is not None:
        raise ValueError('The stop argument has to be None if the value of start '
                         'is a list.')
    elif isinstance(arrays, list):
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [None if x is None else x[start] for x in arrays]
        else:
            if len(arrays) == 1:
                return arrays[0][start:stop]
            return [None if x is None else x[start:stop] for x in arrays]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return arrays[start]
        elif hasattr(start, '__getitem__'):
            return arrays[start:stop]
        else:
            return [None]


def get_param_sum(model):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    logging.info("The total number of parameters：" + str(k))


def log(dataset, model, strategy):
    result_save_dir = os.path.join('../result', dataset, time.strftime("%Y-%m-%d", time.localtime()))
    if not os.path.exists(result_save_dir):
        os.mkdir(result_save_dir)
    filename = str(model) + '_' + strategy + '_result_' + str(time.strftime("%H-%M-%S", time.localtime())) + '.log'
    logging.basicConfig(filename=os.path.join(result_save_dir, filename), level=logging.INFO, filemode='w')


def random_selected_interaction_type(pair_feature_len):
    selected_interaction_type = np.random.rand(pair_feature_len)
    for i in range(pair_feature_len):
        if selected_interaction_type[i] < 0.25:
            selected_interaction_type[i] = 0
        elif selected_interaction_type[i] < 0.5:
            selected_interaction_type[i] = 1
        elif selected_interaction_type[i] < 0.75:
            selected_interaction_type[i] = 2
        else :
            selected_interaction_type[i] = 3
    return torch.tensor(np.array(selected_interaction_type, dtype=int))