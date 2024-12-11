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
from utils.utils import tcl_depth_index, get_combined_do
import random
import sys


class MultiDimEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, target_shape):
        super(MultiDimEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.target_shape = target_shape
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        embedded = self.embedding(x)  # (batch_size, *, embedding_dim)
        target_dims = (-1,) + self.target_shape
        embedded = embedded.view(*embedded.shape[:-1], *target_dims)
        return embedded

class FM_Model(BaseModel):
    def __init__(self, feature_columns, selected_interaction_type,
                 param_save_dir, embedding_size=20, mutation=True,
                 mutation_probability=0.5,
                 activation='tanh', seed=1024, device='cpu'):
        super(FM_Model, self).__init__()
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
        self.rnn1 = torch.nn.LSTM(
            dnn_hidden_units[-1] * 2, dnn_hidden_units[-1] * 2, device=self.device)
        self.rnn2 = torch.nn.LSTM(
            dnn_hidden_units[-1] * 2, dnn_hidden_units[-1] * 2, device=self.device)

        # self.dropout = nn.Dropout(FM_Config['FM']['drop_out'])  # Dropout 层，设置为 0.5 的概率

        self.dnn_linear_1 = nn.Linear(
            dnn_hidden_units[-1] * 2, 1, bias=False).to(device)
        self.dnn_linear_2 = nn.Linear(
            dnn_hidden_units[-1] * 2, 1, bias=False).to(device)

    def forward(self, x, mutation=False):
        ### DO
        batch_size, seq_length, _ = x.shape

        linear_out = self.linear(x, True)
        linear_out = self.dnn_alpha(linear_out)

        # deep part
        interation_out = self.interaction_operation(x, mutation)
        interation_out = self.dnn_beta(interation_out)

        interation_out = interation_out.view(
            batch_size, seq_length, interation_out.shape[-1])

        lstm_input = torch.cat((linear_out, interation_out), dim=-1)

        lstm_output1, (h_n, c_n) = self.rnn1(lstm_input)

        lstm_output2, (h_n, c_n) = self.rnn2(lstm_input)


        # logit = self.dnn_linear(lstm_output)
        output_1 = self.dnn_linear_1(lstm_output1)
        output_2 = self.dnn_linear_2(lstm_output2)

        # output = torch.cat((output_1.unsqueeze(-1), output_2.unsqueeze(-1)), dim=-1)
        output_1 = F.relu(output_1)
        output_2 = F.relu(output_2)
        
        return output_1, output_2

    def before_train(self, trainConfig=None):
        self.metrics_names = ["loss"]
        all_parameters = self.parameters()
        structure_params = {self.linear.alpha, self.interaction_operation.beta}
        net_params = [i for i in all_parameters if i not in structure_params]
        self.structure_optim = self.get_structure_optim(structure_params)

        if trainConfig:
            lr = trainConfig['learning_rate']
            self.net_optim = self.get_net_optim(net_params, lr)
        else:
            self.net_optim = self.get_net_optim(net_params)
        self.loss_func = F.mse_loss
        # self.metrics = self.get_metrics(["mse_loss", "mae", "rmse", "mape"])
        self.metrics = self.get_metrics(["mse_loss", "mae", "rmse"])

    def get_net_optim(self, net_params, lr=None):
        if lr:
            optimizer = optim.Adam(net_params, lr=lr)
        else:
            optimizer = optim.Adam(net_params, lr=float(
                FM_Config['FM']['net_optim_lr']))
        return optimizer

    def get_structure_optim(self, structure_params):
        optimizer = gRDA(structure_params, lr=float(FM_Config['FM']['gRDA_optim_lr']),
                         c=FM_Config['FM']['c'], mu=FM_Config['FM']['mu'])
        return optimizer

    def get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "mse_loss":
                    if set_eps:
                        metrics_[metric] = self._mse_loss
                    else:
                        metrics_[metric] = mean_squared_error
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mae":
                    metrics_[metric] = mean_absolute_error
                # if metric == "mape":
                #     metrics_[metric] = mean_absolute_percentage_error
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "rmse":
                    metrics_[metric] = partial(
                        mean_squared_error, squared=False)
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                        y_true, np.where(y_pred > 0.5, 1, 0))
                self.metrics_names.append(metric)
        return metrics_

    def new(self):
        pair_feature_len = len(
            self.interaction_operation.selected_interaction_type)
        random_type = random_selected_interaction_type(pair_feature_len)
        model_new = FM_Model(feature_columns=self.feature_columns,
                               param_save_dir=self.param_save_dir,
                               selected_interaction_type=random_type,
                               mutation=self.mutation,
                               mutation_probability=self.mutation_probability,
                               embedding_size=self.embedding_size, device=self.device)

        return model_new.to(self.device)

    def replace(self, new_model):
        self.load_state_dict(new_model.state_dict())
        self.interaction_operation.selected_interaction_type = new_model.interaction_operation.selected_interaction_type

    def fit_n_plus_1(self, x=None, y=None, val_x=None, val_y=None, batch_size=None, epochs=1, initial_epoch=0,
                     shuffle=True):

        do_validation = False
        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(x),
            torch.from_numpy(y))
        if batch_size is None:
            batch_size = 4
        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        model = self.train()
        loss_func = self.loss_func
        net_optim = self.net_optim
        structure_optim = self.structure_optim
        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        print("Cognitive EvoLutionary Search period")
        print("Train on {0} samples, {1} steps per epoch".format(
            len(train_tensor_data), steps_per_epoch))


        index_by_all = 0
        for epoch in range(initial_epoch, epochs):
            print(f"----Epoch:{epoch}----")
            model = self.train()
            epoch_start_time = time.time()
            epoch_logs = {}
            # total_loss_epoch = 0
            train_result = {}
            if epoch % 5 == 0:
                torch.save(model.state_dict(), os.path.join(
                    self.param_save_dir, f'model_epoch_{epoch}' + '.pth'))


            parent_num = self.population_size
            parent_models = [self.new() for x in range(parent_num)]
            parent_loss = [float('inf') for x in range(parent_num)]
            child_replace_parent_count = 0
            try:
                with tqdm(enumerate(train_loader)) as t:
                    for index, (x_train, y_train) in t:
                        # if index % 100 == 0:
                        #     self.after_train(
                        #         self.param_save_dir, name='round' + str(int(index / 100)))
                        if index_by_all % 100 == 0:
                            self.after_train(
                                self.param_save_dir, name='round' + str(int(index_by_all)))
                        

                        index_by_all += 1
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()
                        if index % self.mutation_step_size == 0:
                            # update parent models and parent loss only if the current model is better than one of the parent models

                            current_y_pred1, current_y_pred2 = model(
                                x, mutation=False)  # 1: DO, 2:Temperature
                            current_loss_1 = loss_func(
                                current_y_pred1.squeeze(), y[:, :, 0].squeeze(), reduction='sum')
                            current_loss_2 = loss_func(
                                current_y_pred2.squeeze(), y[:, :, 1].squeeze(), reduction='sum')
                            current_l1_loss = calculate_l1_loss(model)
                            current_smoothness_loss = calculate_smoothness_loss(current_y_pred2.squeeze())
                            current_loss = FM_Config['FM']['lambda_1'] * \
                                current_loss_1 + \
                                FM_Config['FM']['lambda_2'] * \
                                current_loss_2 + \
                                FM_Config['FM']['lambda_l1'] * current_l1_loss + \
                                FM_Config['FM']['smooth_loss'] * current_smoothness_loss

                            max_loss = max(parent_loss)
                            if current_loss < max_loss:
                                worst_parent_index = parent_loss.index(
                                    max_loss)
                                parent_models[worst_parent_index].replace(self)
                                parent_loss[worst_parent_index] = current_loss
                                child_replace_parent_count += 1

                            # adapt the mutation_probability accroding to the 1/5 successful rule
                            if index % (self.mutation_step_size * self.adaptation_step_size) == 0 and index != 0:
                                # self.replace_count(self.param_save_dir, child_replace_parent_count)
                                if child_replace_parent_count < 2:
                                    self.interaction_operation.mutation_probability *= 0.99
                                elif child_replace_parent_count > 2:
                                    self.interaction_operation.mutation_probability /= 0.99
                                self.mutation_probability = self.interaction_operation.mutation_probability
                                child_replace_parent_count = 0

                            # apply the crossover mechanism to the parent models
                            self.crossover(parent_models)
                            # muation to generate the offspring model
                            y_pred_1, y_pred_2 = model(x, mutation=True)
                        else:
                            y_pred_1, y_pred_2 = model(x, mutation=True)

                        net_optim.zero_grad()
                        structure_optim.zero_grad()

                        l1_loss = calculate_l1_loss(model)
                        current_smoothness_loss = calculate_smoothness_loss(y_pred_2.squeeze())
                        

                        loss = FM_Config['FM']['lambda_1'] * loss_func(y_pred_1.squeeze(), y[:, :, 0].squeeze(
                        ), reduction='sum') + FM_Config['FM']['lambda_2'] * loss_func(y_pred_2.squeeze(), y[:, :, 1].squeeze(), reduction='sum')
                        + FM_Config['FM']['lambda_l1'] * l1_loss + FM_Config['FM']['smooth_loss'] * current_smoothness_loss

                        y_pred = torch.cat(
                            (y_pred_1.unsqueeze(-1), y_pred_2.unsqueeze(-1)), dim=-1).squeeze()
                        loss.backward()
                        net_optim.step()
                        structure_optim.step()
                        for name, metric_fun in self.metrics.items():
                            if name not in train_result:
                                train_result[name] = []
                            if y.ndim == 3:
                                y = y.view(-1)
                                y_pred = y_pred.view(-1)
                            train_result[name].append(metric_fun(
                                y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))

            except KeyboardInterrupt:
                t.close()
                raise
            t.close()
            # epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result

            epoch_time = int(time.time() - epoch_start_time)
            logging.info('Epoch {0}/{1}'.format(epoch + 1, epochs))

            eval_str = "{0}s".format(epoch_time)

            for name in self.metrics:
                eval_str += " - " + name + \
                            ": {0: .4f}".format(epoch_logs[name])

            if do_validation:
                for name in self.metrics:
                    eval_str += " - " + "val_" + name + \
                                ": {0: .4f}".format(epoch_logs["val_" + name])
            logging.info(eval_str)

    # fit temperature
    def fit_stage2(self, x=None, y=None, unsup_x = None, phy_data = None, hypsography = None, val_x=None, val_y=None, batch_size=None, epochs=1, initial_epoch=0,
                   config=None,  shuffle=True, use_gpu = True):

        do_validation = False
        epochs = config['train_epochs']
        self.loss_func = nn.MSELoss()
        seqLength = x.shape[1]
        self.depth_areas = torch.from_numpy(hypsography).float().flatten()
        self.n_depth = self.depth_areas.shape[0]
        unsup_batch_size = unsup_x.shape[0]

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(x),
            torch.from_numpy(y))
        unsup_tensor_data = Data.TensorDataset(torch.from_numpy(unsup_x), torch.from_numpy(phy_data))
        if batch_size is None:
            batch_size = 4
        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        model = self.train()
        loss_func = self.loss_func
        net_optim = self.net_optim
        structure_optim = self.structure_optim
        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        logging.info('-' * 50)
        logging.info("Train on {0} samples, {1} steps per epoch".format(
            len(train_tensor_data), steps_per_epoch))

        n_batches = sample_num

        manualSeed = [random.randint(1, 99999999) for i in range(epochs)] 
        for epoch in range(epochs):
            if use_gpu:
                torch.cuda.manual_seed(manualSeed[epoch])

            batch_sampler_all = RandomContiguousBatchSampler(unsup_batch_size, seqLength, batch_size=self.n_depth, n_batches = n_batches)
            
            train_loader = DataLoader(
                dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)
            
            phys_loader = DataLoader(
                dataset=unsup_tensor_data, batch_sampler=batch_sampler_all, pin_memory=True, shuffle=False)

            multi_loader = MultiLoader([train_loader, phys_loader])
            model = self.train()
            epoch_start_time = time.time()
            epoch_logs = {}
            total_loss_epoch = 0
            sup_loss_epoch = 0
            unsup_dc_loss_epoch = 0
            unsup_ec_loss_epoch = 0
            batches_done = 0
            train_result = {}
            for i, batches in  tqdm(enumerate(multi_loader)):
                    #load data
                    for j, b in enumerate(batches):
                        if j == 0:
                            inputs, targets = b
                            inputs = inputs.to(self.device).float()
                            targets = targets.to(self.device).float()
                        if j == 1:
                            unsup_inputs, unsup_data = b
                            unsup_inputs = unsup_inputs.to(self.device).float()
                            unsup_data = unsup_data.to(self.device).float()

                    _, y_pred = model(inputs, mutation=False)
                    y_pred = y_pred.squeeze()
                    net_optim.zero_grad()
                    loss_indices = torch.isfinite(targets.squeeze())
                    targets = targets.squeeze()
                    sup_loss = loss_func(y_pred[loss_indices].squeeze(), targets[loss_indices].squeeze())

                    _, unsup_outputs = model(unsup_inputs, mutation=False)
                    unsup_outputs = unsup_outputs.squeeze()
                    dc_unsup_loss = torch.tensor(0).to(self.device).float()
                    if config['dc_lambda'] > 0:
                        dc_unsup_loss = calculate_dc_loss(unsup_outputs, self.n_depth, use_gpu = True)                

                    # print("dc_unsup_loss:", dc_unsup_loss)
                    self.depth_areas = self.depth_areas.to(self.device).float()
                    ec_unsup_loss = torch.tensor(0).to(self.device).float()
                    if config['ec_lambda'] > 0:
                        ec_unsup_loss = calculate_ec_loss_manylakes(unsup_inputs,
                                                unsup_outputs,
                                                unsup_data,
                                                labels=None, #
                                                dates=None,  #                               
                                                depth_areas=self.depth_areas,
                                                n_depths=self.n_depth,
                                                ec_threshold=config['ec_threshold'],
                                                use_gpu = True, 
                                                combine_days=1)
                        
                    l1_loss = calculate_l1_loss(model)
                    # loss = sup_loss + config['ec_lambda'] * ec_unsup_loss + config['dc_lambda'] * dc_unsup_loss + config['lambda1'] * reg1_loss
                    loss = sup_loss + config['ec_lambda'] * ec_unsup_loss + config['dc_lambda'] * dc_unsup_loss +  config['lambda1'] * l1_loss 
                    if torch.isnan(loss):
                        # print(f"Skipping iteration {i} due to NaN loss")
                        continue

                    total_loss_epoch += loss.item()
                    sup_loss_epoch += sup_loss.item()
                    unsup_ec_loss_epoch += ec_unsup_loss.item()
                    unsup_dc_loss_epoch += dc_unsup_loss.item()
                    loss.backward()
                    batches_done += 1
                    net_optim.step()

            epoch_logs["total_loss"] = total_loss_epoch / batches_done
            epoch_logs["sup_loss"] = sup_loss_epoch / batches_done
            epoch_logs["ec_loss"] = unsup_ec_loss_epoch / batches_done
            epoch_logs["dc_loss"] = unsup_dc_loss_epoch / batches_done

            epoch_time = int(time.time() - epoch_start_time)
            logging.info('Epoch {0}/{1}'.format(epoch + 1, epochs))


            eval_str = "{0}s - total_loss: {1:.4f} - sup_loss: {2:.4f} - ec_loss: {3:.4f} - dc_loss: {4:.4f}".format(
                epoch_time, epoch_logs["total_loss"], epoch_logs["sup_loss"], epoch_logs["ec_loss"], epoch_logs["dc_loss"]
            )
            logging.info(eval_str)
    
    # fit Oxygen data
    def fit_stage3(self, x=None, y=None, unsup_x = None, phy_data = None, hypsography = None, val_x=None, val_y=None, batch_size=None, epochs=1, initial_epoch=0,
                   config=None,  shuffle=True, use_gpu = True):

        do_validation = False
        epochs = config['train_epochs']
        self.loss_func = nn.MSELoss()
        seqLength = x.shape[1]
        self.depth_areas = torch.from_numpy(hypsography).float().flatten()
        self.n_depth = self.depth_areas.shape[0]
        unsup_batch_size = unsup_x.shape[0]

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(x),
            torch.from_numpy(y))
        unsup_tensor_data = Data.TensorDataset(torch.from_numpy(unsup_x), torch.from_numpy(phy_data))
        if batch_size is None:
            batch_size = 4
        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        model = self.train()
        loss_func = self.loss_func
        net_optim = self.net_optim
        structure_optim = self.structure_optim
        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        logging.info('-' * 50)
        logging.info("Train on {0} samples, {1} steps per epoch".format(
            len(train_tensor_data), steps_per_epoch))

        n_batches = sample_num

        manualSeed = [random.randint(1, 99999999) for i in range(epochs)] 
        for epoch in range(epochs):
            if use_gpu:
                torch.cuda.manual_seed(manualSeed[epoch])

            batch_sampler_all = RandomContiguousBatchSampler(unsup_batch_size, seqLength, batch_size=self.n_depth, n_batches = n_batches)
            
            train_loader = DataLoader(
                dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)
            
            phys_loader = DataLoader(
                dataset=unsup_tensor_data, batch_sampler=batch_sampler_all, pin_memory=True, shuffle=False)

            multi_loader = MultiLoader([train_loader, phys_loader])
            model = self.train()
            epoch_start_time = time.time()
            epoch_logs = {}
            total_loss_epoch = 0
            sup_loss_epoch = 0
            avg_total_DOC_conservation_loss = 0
            avg_upper_DOC_conservation_loss = 0
            avg_lower_DOC_conservation_loss = 0
            batches_done = 0
            train_result = {}
            for i, batches in  tqdm(enumerate(multi_loader)):
                    #load data
                    for j, b in enumerate(batches):
                        if j == 0:
                            inputs, targets = b
                            inputs = inputs.to(self.device).float()
                            targets = targets.to(self.device).float()
                        if j == 1:
                            unsup_inputs, unsup_data = b
                            unsup_inputs = unsup_inputs.to(self.device).float()
                            unsup_data = unsup_data.to(self.device).float()

                    y_pred, _ = model(inputs, mutation=False)
                    y_pred = y_pred.squeeze()
                    net_optim.zero_grad()
                    loss_indices = torch.isfinite(targets.squeeze())
                    targets = targets.squeeze()
                    sup_loss = loss_func(y_pred[loss_indices].squeeze(), targets[loss_indices].squeeze())

                    unsup_outputs, _ = model(unsup_inputs, mutation=False)
                    # unsup_outputs = unsup_outputs.squeeze()
                    unsup_outputs = unsup_outputs[[0, -1], :, :]
                    flux_data = unsup_data[[0, -1], :, :]

                
                    total_DOC_conservation_loss = torch.tensor(0).to(self.device).float()
                    upper_DOC_conservation_loss = torch.tensor(0).to(self.device).float()
                    lower_DOC_conservation_loss = torch.tensor(0).to(self.device).float()
                    if config['use_unsup']  == 1:
                        index = 0
                        total_DOC_conservation_loss += calculate_total_DOC_conservation_loss(flux_data[index:index+2,:,:], unsup_outputs[index:index+2,:,:], config['doc_threshold'], 1, use_gpu)
                        upper_loss, lower_loss  = calculate_stratified_DOC_conservation_loss(flux_data[index:index+2,:,:], unsup_outputs[index:index+2,:,:], config['doc_threshold'], 1, use_gpu)
                        upper_DOC_conservation_loss += upper_loss
                        lower_DOC_conservation_loss += lower_loss 

                        
                    l1_loss = calculate_l1_loss(model)
                    # loss = sup_loss + config['ec_lambda'] * ec_unsup_loss + config['dc_lambda'] * dc_unsup_loss + config['lambda1'] * reg1_loss
                    loss = sup_loss + config['lambda1'] * l1_loss + config['lambda_total'] * total_DOC_conservation_loss \
                        + config['lambda_stratified_epi'] * upper_DOC_conservation_loss + config['lambda_stratified_hypo'] * lower_DOC_conservation_loss
        
                    if torch.isnan(loss):
                        # print(f"Skipping iteration {i} due to NaN loss")
                        continue

                    total_loss_epoch += loss.item()
                    sup_loss_epoch += sup_loss.item()
                    avg_total_DOC_conservation_loss += total_DOC_conservation_loss.item()
                    avg_upper_DOC_conservation_loss += upper_DOC_conservation_loss.item()
                    avg_lower_DOC_conservation_loss += lower_DOC_conservation_loss .item()
                    loss.backward()
                    batches_done += 1
                    net_optim.step()


            epoch_logs["total_loss"] = total_loss_epoch / batches_done
            epoch_logs["sup_loss"] = sup_loss_epoch / batches_done


            epoch_logs["total_DO_loss"] = avg_total_DOC_conservation_loss / batches_done
            epoch_logs["upper_DO_loss"] = avg_upper_DOC_conservation_loss / batches_done
            epoch_logs["lower_DO_loss"] = avg_lower_DOC_conservation_loss / batches_done

            epoch_time = int(time.time() - epoch_start_time)
            logging.info('Epoch {0}/{1}'.format(epoch + 1, epochs))


            eval_str = "{0}s - total_loss: {1:.4f} - sup_loss: {2:.4f} - total_DO_loss: {3:.4f} - upper_DO_loss: {4:.4f} - lower_DO_loss: {5:.4f}".format(
                epoch_time, 
                epoch_logs["total_loss"], 
                epoch_logs["sup_loss"], 
                epoch_logs["total_DO_loss"], 
                epoch_logs["upper_DO_loss"], 
                epoch_logs["lower_DO_loss"]
            )

            logging.info(eval_str)

    def after_train(self, param_save_dir, name=""):
        state = {'alpha': self.linear.alpha,
                 'beta': self.interaction_operation.beta,
                 }
        if name == "":
            # Saving the final value of alpha and beta weight
            param_save_path = os.path.join(self.param_save_dir,
                                           'alpha_beta-c' + str(FM_Config['FM']['c']) + '-mu' + str(
                                               FM_Config['FM']['mu']) + '-embedding_size' + str(
                                               self.embedding_size) + '.pth')
        else:
            # Saving the current value of alpha and beta weight in the evolution period
            param_save_path = os.path.join(param_save_dir, "evolution", "alpha_beta",
                                           'alpha_beta-c' + str(FM_Config['FM']['c']) + '-mu' + str(
                                               FM_Config['FM']['mu']) + '-embedding_size' + str(
                                               self.embedding_size) + "_" + name + '.pth')
        torch.save(state, param_save_path)

        selected_interaction_type = self.interaction_operation.selected_interaction_type
        if name == "":
            # Saving the final value of interaction types
            param_save_file_path = os.path.join(param_save_dir, 'interaction_type-embedding_size-' +
                                                str(self.embedding_size) + '.pkl')
        else:
            # Saving the current value of interaction types in the evolution period
            param_save_file_path = os.path.join(param_save_dir, "evolution", "operation_type",
                                                'interaction_type-embedding_size-' +
                                                str(self.embedding_size) + "_" + name + '.pkl')
        with open(param_save_file_path, 'wb') as f:
            pkl.dump(selected_interaction_type, f)

        # Saving the value of mutation_probability in the evolution period
        mutation_probability_save_file_path = os.path.join(
            param_save_dir, 'mutation_probability' + '.txt')
        with open(mutation_probability_save_file_path, 'a') as f:
            f.write(str(self.mutation_probability) + '\n')

        # save LSTM param
        if name == "" and FM_Config['FM']['save_lstm_param']:
            lstm_param = self.rnn1.state_dict()
            torch.save(lstm_param, os.path.join(
                param_save_dir, 'lstm_param1' + '.pth'))

        if name == "" and FM_Config['FM']['save_lstm_param']:
            lstm_param = self.rnn2.state_dict()
            torch.save(lstm_param, os.path.join(
                param_save_dir, 'lstm_param2' + '.pth'))

        if name == "" and FM_Config['FM']['save_lstm_param']:
            dnn_linear_param = self.dnn_linear_1.state_dict()
            torch.save(dnn_linear_param, os.path.join(
                param_save_dir, 'dnn_linear_param1' + '.pth'))

        if name == "" and FM_Config['FM']['save_lstm_param']:
            dnn_linear_param = self.dnn_linear_2.state_dict()
            torch.save(dnn_linear_param, os.path.join(
                param_save_dir, 'dnn_linear_param2' + '.pth'))

        if name == "" and FM_Config['FM']['save_lstm_param']:
            dnn_beta_param = self.dnn_beta.state_dict()
            torch.save(dnn_beta_param, os.path.join(
                param_save_dir, 'dnn_beta_param' + '.pth'))
        if name == "" and FM_Config['FM']['save_lstm_param']:
            dnn_alpha_param = self.dnn_alpha.state_dict()
            torch.save(dnn_alpha_param, os.path.join(
                param_save_dir, 'dnn_alpha_param' + '.pth'))

    def replace_count(self, param_save_dir, count):
        save_file_path = os.path.join(
            param_save_dir, 'child_replace_parent_count' + '.txt')
        with open(save_file_path, 'a') as f:
            f.write(str(count) + '\n')

    def crossover(self, parent_models):
        """
        crossover mechanism: select the fittest operation (of which interaction has the largest relevance (beta)) from the population
        """
        p_model = parent_models[0]
        beta = p_model.interaction_operation.beta
        beta_vstack = beta
        interaction = p_model.interaction_operation.selected_interaction_type
        interaction_vstack = interaction
        interaction_vstack = interaction_vstack.to(self.device)
        for i in range(1, len(parent_models)):
            p_model = parent_models[i]
            beta = p_model.interaction_operation.beta
            beta_vstack = torch.vstack((beta_vstack, beta))
            interaction = p_model.interaction_operation.selected_interaction_type
            interaction = interaction.to(self.device)
            interaction_vstack = torch.vstack(
                (interaction_vstack, interaction))

        max_beta, index = torch.max(beta_vstack, dim=0)
        self.interaction_operation.beta.weight = Parameter(max_beta)
        index = index.unsqueeze(dim=0)
        selected_interaction_type = interaction_vstack.gather(0, index)
        self.interaction_operation.selected_interaction_type = selected_interaction_type.squeeze()


    def get_temp(self, x):
        model = self.eval()
        with torch.no_grad():
            x = torch.from_numpy(x).float()
            _, y_temp = model(x, None)
            return y_temp.squeeze().numpy()
        
    def get_do(self, x):
        model = self.eval()
        with torch.no_grad():
            x = torch.from_numpy(x).float()
            y_do, _ = model(x, None)
            return y_do.squeeze().numpy()

    def predict_temp(self, x, y, phys_data, hypsography, config, dataset, save_path):
        model = self.eval()
        loss_func = nn.MSELoss()
        self.depth_areas = torch.from_numpy(hypsography).float().flatten()
        self.n_depth = self.depth_areas.shape[0]
        x = torch.from_numpy(x)
        y = torch.from_numpy(y).squeeze()
        phys_data = torch.from_numpy(phys_data)
        tensor_data = Data.TensorDataset(x, y, phys_data)
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=self.n_depth)
        pred_ans = []

        batch_done = 0
        sup_loss_all = 0
        dc_unsup_loss_all = 0
        ec_unsup_loss_all = 0
        with torch.no_grad():
            for index, (x_test, y_test, phys_data) in enumerate(test_loader):
                input = x_test.to(self.device).float()
                # target = y_test.to(self.device).float()
                _, y_pred = model(input, None)
                y_pred = y_pred.cpu().squeeze()
                dc_unsup_loss = calculate_dc_loss(y_pred, self.n_depth, use_gpu = False)    
                ec_unsup_loss = calculate_ec_loss_manylakes(x,
                        y_pred,
                        phys_data,
                        labels=None, #
                        dates=None,  #                               
                        depth_areas=self.depth_areas,
                        n_depths=self.n_depth,
                        ec_threshold=config['ec_threshold'],
                        use_gpu = False, 
                        combine_days=1)
                targets = y_test.cpu().float()
                loss_indices = torch.isfinite(targets.squeeze())
                targets = targets.squeeze()
                sup_loss = loss_func(y_pred[loss_indices].squeeze(), targets[loss_indices].squeeze())
                dc_unsup_loss_all += dc_unsup_loss
                ec_unsup_loss_all += ec_unsup_loss
                
                # print("dc_unsup_loss:", dc_unsup_loss)
                # print("ec_unsup_loss:", ec_unsup_loss)
                # print("sup_loss", sup_loss)
                y_pred = y_pred.data.numpy()
                pred_ans.append(y_pred)
                batch_done += 1

        pred_ans = np.concatenate(pred_ans).astype("float64")
        assert y.shape == pred_ans.shape
        y_np = y.cpu().numpy().astype("float64")
        loss_indices = np.isfinite(y_np)
        sup_loss_all = loss_func(torch.from_numpy(pred_ans[loss_indices]), torch.from_numpy(y_np[loss_indices])).item() ** 0.5
        tcl_depth = x[:,:,tcl_depth_index].numpy().squeeze()
        is_stratified = tcl_depth.squeeze()
        is_stratified = np.where(is_stratified == 0, 0, 1)

        winter_mask = (is_stratified == 0)
        summer_mask = (is_stratified == 1)

        winter_loss_indices = loss_indices & winter_mask
        summer_loss_indices = loss_indices & summer_mask
        pred_winter = torch.from_numpy(pred_ans[winter_loss_indices])
        y_winter = torch.from_numpy(y_np[winter_loss_indices])
        pred_summer = torch.from_numpy(pred_ans[summer_loss_indices])
        y_summer = torch.from_numpy(y_np[summer_loss_indices])

        sup_loss_winter = loss_func(pred_winter, y_winter).item() ** 0.5
        sup_loss_summer = loss_func(pred_summer, y_summer).item() ** 0.5

        print("sup_loss_winter:", sup_loss_winter)
        print("sup_loss_summer:", sup_loss_summer)

        pred_save_path = os.path.join(save_path, f'pred_{dataset}.npy')
        y_save_path = os.path.join(save_path, f'obs_{dataset}.npy')

        np.save(pred_save_path, pred_ans)
        np.save(y_save_path, y)

        dc_unsup_loss_all = dc_unsup_loss_all/batch_done
        ec_unsup_loss_all = ec_unsup_loss_all/batch_done

        metric = {'sup_loss_all': sup_loss_all,
                  'sup_loss_winter': sup_loss_winter,
                  'sup_loss_summer' : sup_loss_summer,
                   'dc_unsup_loss': dc_unsup_loss_all,
                   'ec_unsup_loss_all' : ec_unsup_loss_all}
        
        metric_path = os.path.join(save_path, f'metric_{dataset}.npy')
        np.save(metric_path, metric, allow_pickle=True)
        loaded_metric = np.load(metric_path, allow_pickle=True).item()

        print(loaded_metric)

        logging.info(f"-------- Ecaluate {dataset} --------")
        logging.info(f"sup_loss_all: {sup_loss_all}")
        logging.info(f"sup_loss_winter: {sup_loss_winter}")
        logging.info(f"sup_loss_summer: {sup_loss_summer}")
        logging.info(f"dc_unsup_loss: {dc_unsup_loss_all}")
        logging.info(f"ec_unsup_loss_all: {ec_unsup_loss_all}")
        logging.info(f"-------- END --------")
        logging.info("")
        return metric

    def predict_do(self, x, y, phys_data, hypsography, config, dataset, save_path):
        model = self.eval()
        loss_func = nn.MSELoss()
        self.depth_areas = torch.from_numpy(hypsography).float().flatten()
        self.n_depth = self.depth_areas.shape[0]
        x = torch.from_numpy(x)
        y = torch.from_numpy(y).squeeze()
        phys_data = torch.from_numpy(phys_data)
        tensor_data = Data.TensorDataset(x, y, phys_data)
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=self.n_depth)
        pred_ans = []
        pred_raw = []
        targets = []
        tcl_depth = []
        batch_done = 0
        sup_loss_all = 0

        total_DOC_conservation_loss = 0
        upper_DOC_conservation_loss = 0
        lower_DOC_conservation_loss = 0

        if len(test_loader) == 0:
            metric = {
                'sup_loss_all': float('nan'),
                'sup_loss_mixed': float('nan'),
                'sup_loss_epi': float('nan'),
                'sup_loss_hypo': float('nan'),
                'total_DO_loss': float('nan'),
                'upper_DO_loss': float('nan'),
                'lower_DO_loss': float('nan')
            }
            return metric

        with torch.no_grad():
            for index, (x_test, y_test, phys_data) in enumerate(test_loader):
                input = x_test.to(self.device).float()
                # target = y_test.to(self.device).float()
                layer = input[:,:,-1].squeeze().cpu().data.numpy()
                y_pred, _ = model(input, None)
                y_pred = y_pred.cpu().squeeze().data.numpy()
                pred_raw.append(y_pred)
                targets.append(y_test[[0,-1],:])
                y_pred_new = get_combined_do(y_pred, layer)
                pred_ans.append(y_pred_new)
                unsup_outputs =torch.from_numpy(y_pred_new).unsqueeze(2)

                # calculate unsupervised loss
                index = 0
                flux_data = phys_data[[0, -1], :, :]
                total_dc_loss = calculate_total_DOC_conservation_loss(flux_data[index:index+2,:,:], unsup_outputs[index:index+2,:,:], config['doc_threshold'], 1, use_gpu = True)
                upper_loss, lower_loss  = calculate_stratified_DOC_conservation_loss(flux_data[index:index+2,:,:], unsup_outputs[index:index+2,:,:], config['doc_threshold'], 1, use_gpu = True)
                
                total_DOC_conservation_loss += total_dc_loss
                upper_DOC_conservation_loss += upper_loss
                lower_DOC_conservation_loss += lower_loss 

                tcl_depth.append(input[[0,-1],:,tcl_depth_index].cpu().numpy().squeeze())
                batch_done += 1

        pred_ans = np.concatenate(pred_ans).astype("float64")
        pred_raw = np.concatenate(pred_raw).astype("float64")
        targets = np.concatenate(targets).astype("float64")
        tcl_depth = np.concatenate(tcl_depth).astype("float64")
        assert y.shape == pred_raw.shape
        assert targets.shape == pred_ans.shape



        loss_indices = np.isfinite(targets)
        sup_loss_all = loss_func(torch.from_numpy(pred_ans[loss_indices]), torch.from_numpy(targets[loss_indices])).item() ** 0.5
        is_stratified = tcl_depth.squeeze()
        is_stratified = np.where(is_stratified == 0, 0, 1)

        winter_mask = (is_stratified == 0)
        summer_mask = (is_stratified == 1)

        winter_loss_indices = loss_indices & winter_mask
        pred_mixed = torch.from_numpy(pred_ans[winter_loss_indices][::2])  # get 0 2 4 ...
        y_mixed = torch.from_numpy(targets[winter_loss_indices][::2])  # get 0 2 4 ...
        sup_loss_mixed = loss_func(pred_mixed, y_mixed).item() ** 0.5


        summer_loss_indices = loss_indices & summer_mask
        pred_epi = torch.from_numpy(pred_ans[summer_loss_indices][::2])  # get 0 2 4 ...
        y_epi = torch.from_numpy(targets[summer_loss_indices][::2])  # get 0 2 4 ... 
        sup_loss_epi = loss_func(pred_epi, y_epi).item() ** 0.5

        pred_hypo = torch.from_numpy(pred_ans[summer_loss_indices][1::2])  # get 1 3 5 ... 
        y_hypo = torch.from_numpy(targets[summer_loss_indices][1::2])  # get 1 3 5 ... 
        sup_loss_hypo = loss_func(pred_hypo, y_hypo).item() ** 0.5


        pred_save_path = os.path.join(save_path, f'pred_{dataset}.npy')
        y_save_path = os.path.join(save_path, f'obs_{dataset}.npy')

        pred_raw_save_path = os.path.join(save_path, f'pred_{dataset}_raw.npy')
        y_raw_save_path = os.path.join(save_path, f'obs_{dataset}_raw.npy')


        np.save(pred_save_path, pred_ans)
        np.save(pred_raw_save_path, pred_raw)
        np.save(y_save_path, targets)
        np.save(y_raw_save_path, y)

        total_DOC_conservation_loss = total_DOC_conservation_loss/batch_done
        upper_DOC_conservation_loss = upper_DOC_conservation_loss/batch_done
        lower_DOC_conservation_loss = lower_DOC_conservation_loss/batch_done


        metric = {'sup_loss_all': sup_loss_all,
                  'sup_loss_mixed': sup_loss_mixed,
                  'sup_loss_epi' : sup_loss_epi,
                   'sup_loss_hypo': sup_loss_hypo,
                   'total_DO_loss' : total_DOC_conservation_loss,
                   'upper_DO_loss' : upper_DOC_conservation_loss,
                   'lower_DO_loss' : lower_DOC_conservation_loss}
        
        metric_path = os.path.join(save_path, f'metric_{dataset}.npy')
        np.save(metric_path, metric, allow_pickle=True)
        loaded_metric = np.load(metric_path, allow_pickle=True).item()

        print(loaded_metric)

        logging.info(f"-------- Ecaluate {dataset} --------")
        logging.info(f"sup_loss_all: {sup_loss_all}")
        logging.info(f"sup_loss_mixed: {sup_loss_mixed}")
        logging.info(f"sup_loss_epi: {sup_loss_epi}")
        logging.info(f"sup_loss_hypo: {sup_loss_hypo}")
        logging.info(f"total_DO_loss: {total_DOC_conservation_loss}")
        logging.info(f"upper_DO_loss: {upper_DOC_conservation_loss}")
        logging.info(f"lower_DO_loss: {lower_DOC_conservation_loss}")
        logging.info(f"-------- END --------")
        logging.info("")
        return metric

    