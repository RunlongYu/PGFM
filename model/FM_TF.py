import os
import time
import torch
import logging
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import DataLoader
from tqdm import tqdm
from optimizer.gRDA import gRDA
from .baseModel import BaseModel
from .FM_Model import FM_Model
from layer.interactionLayer import InteractionLayer
from layer.linearLayer import NormalizedWeightedLinearLayer
from utils.function_utils import generate_pair_index, slice_arrays
from sklearn.metrics import *
from config.configs import FM_Config, General_Config
import pickle as pkl
from utils.function_utils import random_selected_interaction_type
from torch.nn.parameter import Parameter
from functools import partial
from layer.mlpLayer import DNN
from utils.utils_stage2 import ContiguousBatchSampler, RandomContiguousBatchSampler, MultiLoader
from utils.utils_stage2 import calculate_ec_loss_manylakes, calculate_dc_loss, calculate_l1_loss, calculate_smoothness_loss, calculate_total_DOC_conservation_loss, calculate_stratified_DOC_conservation_loss
import torch.nn as nn
from utils.utils import tcl_depth_index
import random
import sys
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class FM_TF(FM_Model):
    def __init__(self, feature_columns, selected_interaction_type,
                 param_save_dir, embedding_size=20, mutation=True,
                 mutation_probability=0.5,
                 activation='tanh', seed=1024, device='cpu'):
        super(FM_TF, self).__init__(feature_columns, selected_interaction_type, param_save_dir)
        self.feature_columns = feature_columns
        self.bucket_len = len(self.feature_columns)
        self.embedding_size = embedding_size
        self.device = device
        if device == 'cpu':
            torch.manual_seed(seed)
        else:
            torch.cuda.manual_seed(seed)
        self.register_buffer('pair_indexes',
                             torch.tensor(generate_pair_index(len(self.feature_columns), 2)))
        self.interaction_pair_number = len(self.pair_indexes[0])
        self.feature_num = len(self.feature_columns)

        self.param_save_dir = param_save_dir
        self.mutation = mutation
        self.mutation_probability = mutation_probability

        self.interaction_fc_output_dim = int(
            FM_Config['FM']['interaction_fc_output_dim'])
        self.mutation_threshold = float(
            FM_Config['FM']['mutation_threshold'])
        self.mutation_step_size = int(
            FM_Config['FM']['mutation_step_size'])
        self.adaptation_hyperparameter = float(
            FM_Config['FM']['adaptation_hyperparameter'])
        self.adaptation_step_size = int(
            FM_Config['FM']['adaptation_step_size'])
        self.population_size = int(FM_Config['FM']['population_size'])

        # \alpha
        self.linear = NormalizedWeightedLinearLayer(feature_columns=feature_columns,
                                                    alpha_activation=activation, use_alpha=True,
                                                    device=device)

        # \beta
        self.interaction_operation = InteractionLayer(input_dim=embedding_size, feature_columns=feature_columns,
                                                      use_beta=True,
                                                      interaction_fc_output_dim=self.interaction_fc_output_dim,
                                                      selected_interaction_type=selected_interaction_type,
                                                      mutation_threshold=self.mutation_threshold,
                                                      mutation_probability=self.mutation_probability,
                                                      device=device, reduce_sum=False)

        dnn_hidden_units = FM_Config['FM']['dnn_hidden_units']
        self.dnn_beta = DNN(self.interaction_pair_number * self.interaction_fc_output_dim, dnn_hidden_units,
                            dropout_rate=0, use_bn=False)

        self.dnn_alpha = DNN(self.feature_num, [dnn_hidden_units[-1]],
                             dropout_rate=0, use_bn=False)
        # self.dnn_linear = nn.Linear(
        #     dnn_hidden_units[-1] * 2, 1, bias=False).to(device)s
        transformer_layer_1 = TransformerEncoderLayer(d_model=dnn_hidden_units[-1] * 2, nhead=8)
        transformer_layer_2 = TransformerEncoderLayer(d_model=dnn_hidden_units[-1] * 2, nhead=8)
        self.transformer_encoder_1 = TransformerEncoder(transformer_layer_1, num_layers=2)
        self.transformer_encoder_2 = TransformerEncoder(transformer_layer_2, num_layers=2)


        # self.dropout = nn.Dropout(FM_Config['FM']['drop_out'])  # Dropout 层，设置为 0.5 的概率

        self.dnn_linear_1 = nn.Linear(
            dnn_hidden_units[-1] * 2, 1, bias=False).to(device)
        self.dnn_linear_2 = nn.Linear(
            dnn_hidden_units[-1] * 2, 1, bias=False).to(device)
        



    def forward(self, x, mutation=False):
        batch_size, seq_length, _ = x.shape

        linear_out = self.linear(x, True)
        linear_out = self.dnn_alpha(linear_out)

        # deep part
        interaction_out = self.interaction_operation(x, mutation)
        interaction_out = self.dnn_beta(interaction_out)

        interaction_out = interaction_out.view(
            batch_size, seq_length, interaction_out.shape[-1])

        transformer_input = torch.cat((linear_out, interaction_out), dim=-1)

        # Ensure all tensors have the same dtype
        transformer_input = transformer_input.to(torch.float32)

        # Apply Transformers
        transformer_output_1 = self.transformer_encoder_1(transformer_input)
        transformer_output_2 = self.transformer_encoder_2(transformer_input)

        # Flatten the output for the linear layers
        transformer_output_1 = transformer_output_1.view(batch_size * seq_length, -1)
        transformer_output_2 = transformer_output_2.view(batch_size * seq_length, -1)

        # Linear layers
        output_1 = self.dnn_linear_1(transformer_output_1)
        output_2 = self.dnn_linear_2(transformer_output_2)

        # Reshape the output back to the original shape
        output_1 = output_1.view(batch_size, seq_length, -1)
        output_2 = output_2.view(batch_size, seq_length, -1)

        output_1 = F.relu(output_1)
        output_2 = F.relu(output_2)

        return output_1, output_2
    
    def new(self):
        pair_feature_len = len(
            self.interaction_operation.selected_interaction_type)
        random_type = random_selected_interaction_type(pair_feature_len)
        model_new = FM_TF(feature_columns=self.feature_columns,
                               param_save_dir=self.param_save_dir,
                               selected_interaction_type=random_type,
                               mutation=self.mutation,
                               mutation_probability=self.mutation_probability,
                               embedding_size=self.embedding_size, device=self.device)

        return model_new.to(self.device)
