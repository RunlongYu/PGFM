import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
import torch.nn.functional as F
import logging
import torch.utils.data as Data
from torch.utils.data import DataLoader
import random
from utils.utils_stage2 import ContiguousBatchSampler, RandomContiguousBatchSampler, MultiLoader
from utils.utils_stage2 import calculate_ec_loss_manylakes, calculate_dc_loss, calculate_l1_loss, calculate_smoothness_loss, calculate_total_DOC_conservation_loss, calculate_stratified_DOC_conservation_loss
import time
from tqdm import tqdm
from utils.utils import tcl_depth_index, get_combined_do
import numpy as np
import torch.optim as optim
import os


class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, num_layers, hidden_size, dropout=0.1, device = None):
        super(TransformerModel, self).__init__()
        self.input_size = input_size
        self.model_type = 'Transformer'
        self.device = device
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(input_size, 1)  # Assuming 1 output feature

    def forward(self, src):
        src = src.to(self.device)
        src = src.permute(1, 0, 2)  # Transformer expects input of shape (S, N, E)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        output = output.permute(1, 0, 2)  # Back to (N, S, E)
        output = F.relu(output)
        return output

    def before_train(self,trainConfig):
        lr = trainConfig['learning_rate']
        self.loss_func = nn.MSELoss()
        return
    
    def fit_stage2(self, x=None, y=None, unsup_x = None, phy_data = None, hypsography = None, val_x=None, val_y=None, batch_size=None, epochs=1, initial_epoch=0,
                    config=None,  shuffle=True, use_gpu = True):

            do_validation = False
            model = self.train()
            epochs = config['train_epochs']
            self.net_optim = optim.Adam(model.parameters(), lr=config['learning_rate']) 
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

            
            loss_func = self.loss_func
            net_optim = self.net_optim
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

                        y_pred = model(inputs)
                        y_pred = y_pred.squeeze()
                        net_optim.zero_grad()
                        loss_indices = torch.isfinite(targets.squeeze())
                        targets = targets.squeeze()
                        sup_loss = loss_func(y_pred[loss_indices].squeeze(), targets[loss_indices].squeeze())

                        unsup_outputs = model(unsup_inputs)
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

    def fit_stage3(self, x=None, y=None, unsup_x = None, phy_data = None, hypsography = None, val_x=None, val_y=None, batch_size=None, epochs=1, initial_epoch=0,
                   config=None,  shuffle=True, use_gpu = True):

        do_validation = False
        model = self.train()
        epochs = config['train_epochs']
        self.net_optim = optim.Adam(model.parameters(), lr=config['learning_rate']) 
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


                    y_pred = model(inputs)
                    y_pred = y_pred.squeeze()
                    net_optim.zero_grad()
                    loss_indices = torch.isfinite(targets.squeeze())
                    targets = targets.squeeze()
                    sup_loss = loss_func(y_pred[loss_indices].squeeze(), targets[loss_indices].squeeze())

                    unsup_outputs = model(unsup_inputs)
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
                y_pred = model(input)
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
                y_pred = model(input)
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
        pred_mixed = torch.from_numpy(pred_ans[winter_loss_indices][::2])
        y_mixed = torch.from_numpy(targets[winter_loss_indices][::2]) 
        sup_loss_mixed = loss_func(pred_mixed, y_mixed).item() ** 0.5

        summer_loss_indices = loss_indices & summer_mask
        pred_epi = torch.from_numpy(pred_ans[summer_loss_indices][::2])  
        y_epi = torch.from_numpy(targets[summer_loss_indices][::2])  
        sup_loss_epi = loss_func(pred_epi, y_epi).item() ** 0.5

        pred_hypo = torch.from_numpy(pred_ans[summer_loss_indices][1::2])  
        y_hypo = torch.from_numpy(targets[summer_loss_indices][1::2])  
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