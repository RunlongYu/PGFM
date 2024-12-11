import random
import math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import datetime
from datetime import date
from utils import phys_operations
from utils.utils import tcl_depth_index
import pandas as pd
from datetime import date
from math import inf
import sys
import logging
import os
import torch.utils.data as Data


cluster_range = [ 1, 2, 3, 4 ]
# cluster_range = [1]

# random_seed = [40, 42, 44]
random_seed = [44]

# model_type_range = ['lstm', 'pg', 'ecopg']
model_type_range = ['lstm']

dataset_type_list = ['trn', 'val', 'tst']

def getPbDoResults(x, sim_y_pred, y, phys_data, hypsography, config, dataset, save_path):
    loss_func = nn.MSELoss()
    depth_areas = torch.from_numpy(hypsography).float().flatten()
    n_depth = depth_areas.shape[0]
    x = torch.from_numpy(x)
    y = torch.from_numpy(y).squeeze()
    sim_y_pred = torch.from_numpy(sim_y_pred).squeeze()
    phys_data = torch.from_numpy(phys_data)
    tensor_data = Data.TensorDataset(x, sim_y_pred, y, phys_data)
    test_loader = DataLoader(
        dataset=tensor_data, shuffle=False, batch_size=n_depth)
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
        for index, (x_test, sim_y_pred, y_test, phys_data) in enumerate(test_loader):
            input = x_test.float()
            # target = y_test.to(self.device).float()
            layer = input[:,:,-1].squeeze().cpu().data.numpy()
     
            y_pred = sim_y_pred
            y_pred = y_pred.cpu().squeeze().data.numpy()
            pred_raw.append(y_pred)
            targets.append(y_test[[0,-1],:])
            y_pred_new = y_pred[[0,-1],:]
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

def getPbTemperatureResults(x, sim_pred_y, y, phys_data, hypsography, config, dataset, save_path):
        loss_func = nn.MSELoss()
        depth_areas = torch.from_numpy(hypsography).float().flatten()
        n_depth = depth_areas.shape[0]
        x = torch.from_numpy(x)

        sim_pred_y = torch.from_numpy(sim_pred_y).squeeze()
        y = torch.from_numpy(y).squeeze()
        phys_data = torch.from_numpy(phys_data)
        tensor_data = Data.TensorDataset(sim_pred_y, y, phys_data)
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=n_depth)
        pred_ans = []

        batch_done = 0
        sup_loss_all = 0
        dc_unsup_loss_all = 0
        ec_unsup_loss_all = 0
        with torch.no_grad():
            for index, (y_sim_pred, y_test, phys_data) in enumerate(test_loader):
                y_pred = y_sim_pred
                y_pred = y_pred.cpu().squeeze()
                dc_unsup_loss = calculate_dc_loss(y_pred, n_depth, use_gpu = False)    
                ec_unsup_loss = calculate_ec_loss_manylakes(x,
                        y_pred,
                        phys_data,
                        labels=None, #
                        dates=None,  #                               
                        depth_areas=depth_areas,
                        n_depths=n_depth,
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

#Dataset classes
class TemperatureTrainDataset(Dataset):
    #training dataset class, allows Dataloader to load both input/target
    def __init__(self, trn_data):
        # depth_data = depth_trn
        self.len = trn_data.shape[0]
        self.x_data = trn_data[:,:,:-1].float()
        self.y_data = trn_data[:,:,-1].float()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class TotalModelOutputDataset(Dataset):
    #dataset for unsupervised input(in this case all the data)
    def __init__(self, all_data, all_phys_data,all_dates):
        #data of all model output, and corresponding unstandardized physical quantities
        #needed to calculate physical loss
        self.len = all_data.shape[0]
        self.data = all_data[:,:,:-1].float()
        self.label = all_data[:,:,-1].float() #DO NOT USE IN MODEL
        self.phys = all_phys_data.float()
        helper = np.vectorize(lambda x: date.toordinal(pd.Timestamp(x).to_pydatetime()))
        dates = helper(all_dates)
        self.dates = dates

    def __getitem__(self, index):
        return self.data[index], self.phys[index], self.dates[index], self.label[index]

    def __len__(self):
        return self.len

class ContiguousBatchSampler(object):
    def __init__(self, batch_size, n_batches):
        # print("batch size", batch_size)
        # print("n batch ", n_batches)
        self.sampler = torch.randperm(n_batches)
        self.batch_size = batch_size

    def __iter__(self):
        for idx in self.sampler:
            yield torch.arange(idx*self.batch_size, (idx+1)*self.batch_size, dtype=torch.long)

    def __len__(self):
        return len(self.sampler) // self.batch_size

class RandomContiguousBatchSampler(object):
    def __init__(self, n_dates, seq_length, batch_size, n_batches):
        # note: batch size = n_depths here
        #       n_dates = number of all* sequences (*yhat driver dataset)
        # high = math.floor((n_dates-seq_length)/batch_size) dated and probably wrong
        high = math.floor(n_dates/batch_size)

        self.sampler = torch.randint_like(torch.empty(n_batches), low=0, high=high)        
        self.batch_size = batch_size

    def __iter__(self):
        for idx in self.sampler:
            # yield torch.arange(idx*self.batch_size, (idx+1)*self.batch_size, dtype=torch.long) #old
            yield torch.arange(idx*self.batch_size, (idx+1)*self.batch_size, dtype=torch.long) 

    def __len__(self):
        return len(self.sampler) // self.batch_size
    
class MultiLoader(object):
  """This class wraps several pytorch DataLoader objects, allowing each time 
  taking a batch from each of them and then combining these several batches 
  into one. This class mimics the `for batch in loader:` interface of 
  pytorch `DataLoader`.
  Args: 
    loaders: a list or tuple of pytorch DataLoader objects
  """
  def __init__(self, loaders):
    self.loaders = loaders

  def __iter__(self):
    return MyIter(self)

  def __len__(self):
    l =  min([len(loader) for loader in self.loaders])
    # print(l)
    return l

  # Customize the behavior of combining batches here.
  def combine_batch(self, batches):
    return batches

class MyIter(object):
  """An iterator."""
  def __init__(self, my_loader):
    self.my_loader = my_loader
    self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]
    # print("init", self.loader_iters)

  def __iter__(self):
    return self

  def __next__(self):
    # When the shortest loader (the one with minimum number of batches)
    # terminates, this iterator will terminates.
    # The `StopIteration` raised inside that shortest loader's `__next__`
    # method will in turn gets out of this `__next__` method.
    # print("next",     print(self.loader_iters))
    
    ### raw
    # batches = [loader_iter.next() for loader_iter in self.loader_iters]

    batches = [next(loader_iter) for loader_iter in self.loader_iters]

    return self.my_loader.combine_batch(batches)
  

def get_ids():
    ids_pd = pd.read_csv(f'../utils/intersection_ids.csv')
    ids = ids_pd['nhdhr_id'].to_list()

    return ids

def calculate_ec_loss_manylakes(inputs, outputs, phys, labels, dates, depth_areas, n_depths, ec_threshold, use_gpu, combine_days):
    #******************************************************
    #description: calculates energy conservation loss
    #parameters: 
        #@inputs: features
        #@outputs: labels
        #@phys: features(not standardized) of sw_radiation, lw_radiation, etc
        #@labels modeled temp (will not used in loss, only for test)
        #@depth_areas: cross-sectional area of each depth
        #@n_depths: number of depths
        #@use_gpu: gpu flag
        #@combine_days: how many days to look back to see if energy is conserved
    #*********************************************************************************
    debug = False

    diff_vec = torch.empty((phys.size()[1]))
    n_dates = phys.size()[1]
    # outputs = labels  
    outputs = outputs.view(outputs.size()[0], outputs.size()[1])
    # print("modeled temps: ", outputs)
    densities = transformTempToDensity(outputs, use_gpu)
    # print("modeled densities: ", densities)

    #for experiment
    if use_gpu:
        densities = densities.cuda()  
        #loop through sets of n_depths

    #calculate lake energy for each timestep

    lake_energies = calculate_lake_energy(outputs[:,:], densities[:,:], depth_areas)

    
    #calculate energy change in each timestep
    lake_energy_deltas = calculate_lake_energy_deltas(lake_energies, combine_days, depth_areas[0])

    lake_energy_deltas = lake_energy_deltas[1:] #### 


    #calculate sum of energy flux into or out of the lake at each timestep
    # print("dates ", dates[0,1:6])
    lake_energy_fluxes = calculate_energy_fluxes_manylakes(phys[0,:,:], outputs[0,:], combine_days)


    ### can use this to plot energy delta and flux over time to see if they line up
    # doy = np.array([datetime.datetime.combine(date.fromordinal(x), datetime.time.min).timetuple().tm_yday  for x in dates[start_index,:]])
    # doy = doy[1:-1]


    # diff_vec = (lake_energy_deltas - 200).abs_()
    
    diff_vec = (lake_energy_deltas - lake_energy_fluxes).abs_()
    # diff_vec = (lake_energy_deltas - lake_energy_fluxes).abs()

    if debug:
        print("lake_energies:", lake_energies)
        print("lake_energy_deltas:", lake_energy_deltas)
        print("lake_energy_fluxes:", lake_energy_fluxes)
        print("ice flag:",phys[0,1:-1,-1])
        print("diff_vec:", diff_vec)

    
    # phys = phys.cpu()
    # diff_vec = diff_vec.cpu()
    # diff_vec = diff_vec[np.where((phys[0,1:-1,-1] == 0))[0]] #only over no-ice period

    mask = (phys[0, 1:-1, -1] == 0)

    # diff_vec
    filtered_diff_vec = diff_vec[mask]

    if use_gpu:
        phys = phys.cuda()
        filtered_diff_vec = filtered_diff_vec.cuda()
    #actual ice
    # diff_vec = diff_vec[np.where((phys[1:(n_depths-diff_vec.size()[0]-1),9] == 0))[0]]
    # #compute difference to be used as penalty

    mean_value = filtered_diff_vec.mean()
    if debug:
        print("diff_vec.mean():", mean_value)
    if filtered_diff_vec.size() == torch.Size([0]):
        print("diff_vec empty")
        return 0
    else:
        mean_value_rounded = torch.round(mean_value * 10) / 10  # 保留小数
        res = torch.clamp(mean_value - ec_threshold, min=0.0)  
        return res

def calculate_dc_loss(outputs, n_depths, use_gpu):#
    #calculates depth-density consistency loss
    #parameters:
        #@outputs: labels = temperature predictions, organized as depth (rows) by date (cols)
        #@n_depths: number of depths
        #@use_gpu: gpu flag

    assert outputs.size()[0] == n_depths

    densities = transformTempToDensity(outputs, use_gpu)

    # We could simply count the number of times that a shallower depth (densities[:-1])
    # has a higher density than the next depth below (densities[1:])
    # num_violations = (densities[:-1] - densities[1:] > 0).sum()

    # But instead, let's use sum(sum(ReLU)) of the density violations,
    # per Karpatne et al. 2018 (https://arxiv.org/pdf/1710.11431.pdf) eq 3.14
    sum_violations = (densities[:-1] - densities[1:]).clamp(min=0).sum()

    return sum_violations

def calculate_ec_loss(inputs, outputs, phys, labels, dates, depth_areas, n_depths, ec_threshold, use_gpu, combine_days=1):
    #******************************************************
    #description: calculates energy conservation loss
    #parameters: 
        #@inputs: features
        #@outputs: labels
        #@phys: features(not standardized) of sw_radiation, lw_radiation, etc
        #@labels modeled temp (will not used in loss, only for test)
        #@depth_areas: cross-sectional area of each depth
        #@n_depths: number of depths
        #@use_gpu: gpu flag
        #@combine_days: how many days to look back to see if energy is conserved (obsolete)
    #*********************************************************************************

    n_sets = math.floor(inputs.size()[0] / n_depths)#sets of depths in batch
    diff_vec = torch.empty((inputs.size()[1]))
    n_dates = inputs.size()[1]

    
    outputs = outputs.view(outputs.size()[0], outputs.size()[1])
    # print("modeled temps: ", outputs)
    densities = transformTempToDensity(outputs, use_gpu)
    # print("modeled densities: ", densities)


    #for experiment
    if use_gpu:
        densities = densities.cuda()  
    diff_per_set = torch.empty(n_sets) 
    for i in range(n_sets):
        #loop through sets of n_depths

        #indices
        start_index = (i)*n_depths
        end_index = (i+1)*n_depths


        #assert have all depths
        # assert torch.unique(inputs[:,0,1]).size()[0] == n_depths
        # assert torch.unique(inputs[:,100,1]).size()[0] == n_depths
        # assert torch.unique(inputs[:,200,1]).size()[0] == n_depths
        # assert torch.unique(phys[:,0,1]).size()[0] == n_depths
        # assert torch.unique(phys[:,100,1]).size()[0] == n_depths
        # assert torch.unique(phys[:,200,1]).size()[0] == n_depths


        #calculate lake energy for each timestep
        lake_energies = calculate_lake_energy(outputs[start_index:end_index,:], densities[start_index:end_index,:], depth_areas)
        #calculate energy change in each timestep
        lake_energy_deltas = calculate_lake_energy_deltas(lake_energies, combine_days, depth_areas[0])
        lake_energy_deltas = lake_energy_deltas[1:]
        #calculate sum of energy flux into or out of the lake at each timestep
        # print("dates ", dates[0,1:6])
        lake_energy_fluxes = calculate_energy_fluxes(phys[start_index,:,:], outputs[start_index,:], combine_days)
        ### can use this to plot energy delta and flux over time to see if they line up
        doy = np.array([datetime.datetime.combine(date.fromordinal(x), datetime.time.min).timetuple().tm_yday  for x in dates[start_index,:]])
        doy = doy[1:-1]
        diff_vec = (lake_energy_deltas - lake_energy_fluxes).abs_()
        
        # mendota og ice guesstimate
        # diff_vec = diff_vec[np.where((doy[:] > 134) & (doy[:] < 342))[0]]

        #actual ice
        diff_vec = diff_vec[np.where((phys[0,1:-1,9] == 0))[0]]
        # #compute difference to be used as penalty
        if diff_vec.size() == torch.Size([0]):
            diff_per_set[i] = 0
        else:
            diff_per_set[i] = diff_vec.mean()
    if use_gpu:
        diff_per_set = diff_per_set.cuda()
    diff_per_set = torch.clamp(diff_per_set - ec_threshold, min=0)
    print(diff_per_set.mean())
    return diff_per_set.mean()


def calculate_smoothness_loss(y_pred):
    diff = y_pred[:, 1:] - y_pred[:, :-1]
    loss = torch.mean(diff ** 2)
    return loss

def calculate_l1_loss(model):
    def l1_loss(x):
        return torch.abs(x).sum()

    to_regularize = []
    # for name, p in model.named_parameters():
    for name, p in model.named_parameters():
        if 'bias' in name:
            continue
        else:
            #take absolute value of weights and sum
            to_regularize.append(p.view(-1))
    l1_loss_val = torch.tensor(1, requires_grad=True, dtype=torch.float32)
    l1_loss_val = l1_loss(torch.cat(to_regularize))
    return l1_loss_val

def calculate_lake_energy(temps, densities, depth_areas):
    #calculate the total energy of the lake for every timestep
    #sum over all layers the (depth cross-sectional area)*temp*density*layer_height)
    #then multiply by the specific heat of water 
    dz = 0.5 #thickness for each layer, hardcoded for now
    cw = 4186.0 #specific heat of water
    energy = torch.empty_like(temps[0,:])
    n_depths = depth_areas.size()[0]
    depth_areas = depth_areas.view(n_depths,1).expand(n_depths, temps.size()[1])
    energy = torch.sum(depth_areas*temps*densities*0.5*cw, 0)
    
    return energy

def calculate_lake_energy_deltas(energies, combine_days, surface_area):
    #given a time series of energies, compute and return the differences
    # between each time step, or time step interval (parameter @combine_days)
    # as specified by parameter @combine_days
    energy_deltas = torch.empty_like(energies[0:-combine_days])
    time = 86400.0 #seconds per day
    # surface_area = 39865825
    energy_deltas = (energies[1:] - energies[:-1])/(time*surface_area)
    # for t in range(1, energy_deltas.size()[0]):
    #     energy_deltas[t-1] = (energies[t+combine_days] - energies[t])/(time*surface_area) #energy difference converted to W/m^2
    return energy_deltas

def calculate_energy_fluxes(phys, surf_temps, combine_days):
    # print("surface_depth = ", phys[0:5,1])
    fluxes = torch.empty_like(phys[:-combine_days-1,0])

    time = 86400 #seconds per day
    surface_area = 39865825 

    e_s = 0.985 #emissivity of water, given by Jordan
    alpha_sw = 0.07 #shortwave albedo, given by Jordan Read
    alpha_lw = 0.03 #longwave, albeda, given by Jordan Read
    sigma = 5.67e-8 #Stefan-Baltzmann constant
    R_sw_arr = phys[:-1,2] + (phys[1:,2]-phys[:-1,2])/2
    R_lw_arr = phys[:-1,3] + (phys[1:,3]-phys[:-1,3])/2
    R_lw_out_arr = e_s*sigma*(torch.pow(surf_temps[:]+273.15, 4))
    R_lw_out_arr = R_lw_out_arr[:-1] + (R_lw_out_arr[1:]-R_lw_out_arr[:-1])/2

    air_temp = phys[:-1,4] 
    air_temp2 = phys[1:,4]
    rel_hum = phys[:-1,5]
    rel_hum2 = phys[1:,5]
    ws = phys[:-1, 6]
    ws2 = phys[1:,6]
    t_s = surf_temps[:-1]
    t_s2 = surf_temps[1:]
    E = phys_operations.calculate_heat_flux_latent(t_s, air_temp, rel_hum, ws)
    H = phys_operations.calculate_heat_flux_sensible(t_s, air_temp, rel_hum, ws)
    E2 = phys_operations.calculate_heat_flux_latent(t_s2, air_temp2, rel_hum2, ws2)
    H2 = phys_operations.calculate_heat_flux_sensible(t_s2, air_temp2, rel_hum2, ws2)
    E = (E + E2)/2
    H = (H + H2)/2

    #test
    fluxes = (R_sw_arr[:-1]*(1-alpha_sw) + R_lw_arr[:-1]*(1-alpha_lw) - R_lw_out_arr[:-1] + E[:-1] + H[:-1])


    return fluxes

def calculate_energy_fluxes_manylakes(phys, surf_temps, combine_days):
    fluxes = torch.empty_like(phys[:-combine_days-1,0])

    time = 86400. #seconds per day
    surface_area = 39865825. 

    e_s = 0.985 #emissivity of water, given by Jordan
    alpha_sw = 0.07 #shortwave albedo, given by Jordan Read
    alpha_lw = 0.03 #longwave, albeda, given by Jordan Read
    sigma = 5.67e-8 #Stefan-Baltzmann constant
    R_sw_arr = phys[:-1,1] + (phys[1:,1]-phys[:-1,1])/2.0
    R_lw_arr = phys[:-1,2] + (phys[1:,2]-phys[:-1,2])/2.0
    R_lw_out_arr = e_s*sigma*(torch.pow(surf_temps[:]+273.15, 4.0))
    R_lw_out_arr = R_lw_out_arr[:-1] + (R_lw_out_arr[1:]-R_lw_out_arr[:-1])/2.0

    air_temp = phys[:-1,3] 
    air_temp2 = phys[1:,3]
    rel_hum = phys[:-1,4]
    rel_hum2 = phys[1:,4]
    ws = phys[:-1, 5]
    ws2 = phys[1:,5]
    t_s = surf_temps[:-1]
    t_s2 = surf_temps[1:]
    E = phys_operations.calculate_heat_flux_latent(t_s, air_temp, rel_hum, ws)
    H = phys_operations.calculate_heat_flux_sensible(t_s, air_temp, rel_hum, ws)
    E2 = phys_operations.calculate_heat_flux_latent(t_s2, air_temp2, rel_hum2, ws2)
    H2 = phys_operations.calculate_heat_flux_sensible(t_s2, air_temp2, rel_hum2, ws2)
    E = (E + E2)/2
    H = (H + H2)/2

    fluxes = (R_sw_arr[:-1]*(1-alpha_sw) + R_lw_arr[:-1]*(1-alpha_lw) - R_lw_out_arr[:-1] + E[:-1] + H[:-1]) 
    fluxes = fluxes
    return fluxes


def find_stratified_segments(tensor):
    stratified_segments = []
    start = None

    for i, val in enumerate(tensor):
        if val != 0 and start is None:
            start = i
        elif val == 0 and start is not None:
            stratified_segments.append((start, i))
            start = None

    if start is not None:
        stratified_segments.append((start, len(tensor)))

    return stratified_segments

def calculate_stratified_flux(flux_data, t):
    # print("flux shape:", flux_data.shape)
    flux = flux_data.squeeze(0)
    fnep = flux[:, 0]  # simulated net ecosystem production flux
    fmineral = flux[:, 1]  # simulated mineralisation flux( hypo NEP)
    fsed = flux[:, 2]  # simulated net sedimentation flux
    fatm = flux[:, 3]  # simulated atmospheric exchange flux
    fdiff = flux[:, 4]  # simulated diffusion flux
    fentr_epi = flux[:, 5]  # simulated entrainment flux (epilimnion)
    fentr_hyp = flux[:, 6]  # simulated entrainment flux (hypolimnion)

    Flux_epi =  fatm + fnep + fdiff # F_ATM + F_NEP_epi + F_ENTR_epi + F_DIFF_epi  
    Flux_hypo = fmineral - fsed - fdiff # 

    result = torch.cat((Flux_epi.unsqueeze(0), Flux_hypo.unsqueeze(0)), dim=0) # size: [2,350]
    result = result * 0.001

    return result # [2, 350]

def get_Flux_enter(flux_data, t):
    flux = flux_data.squeeze(0)
    fentr_epi = flux[:, 5]  # simulated entrainment flux (epilimnion)
    fentr_hyp = flux[:, 6]  # simulated entrainment flux (hypolimnion)

    Flux_enter = torch.cat((fentr_epi.unsqueeze(0), fentr_hyp.unsqueeze(0)), dim=0) # size: [2,350]
    Flux_enter = Flux_enter*0.001
    return Flux_enter

def calculate_loss_for_segment(segment, V_Both, Flux_Data, Pred_, t, use_gpu):
    debug = False
    Flux = calculate_stratified_flux(Flux_Data, t)
    Flux_enter = get_Flux_enter(Flux_Data, t)
    start = segment[0]
    end = segment[1]
    V_both = V_Both[:,start:end]
    pred = Pred_[:,start:end]
    flux = Flux[:, start:end]
    flux_enter = Flux_enter[:, start:end]
    Do_pre = pred[:,:-1]
    Do_next = pred[:,1:]
    flux = flux[:,1:]
    if debug:
        print("Do_next:", Do_next[1::2,:])
    V_both_pre = V_both[:,:-1]
    V_both_next = V_both[:,1:]
    
    assert torch.all(V_both != 0), "V_total contains zero, which will cause division by zero."

    V_divided = V_both_pre/V_both_next
    if debug:
        print("V_divided shape:", V_divided.shape)

    seg_loss = Do_next - torch.maximum(((Do_pre+flux)*V_divided + flux_enter[:,1:]), torch.tensor(0.0))
    seg_loss = seg_loss.to(Do_pre.dtype)
    if debug:
        print("seg_loss lower", seg_loss[1::2,:])
    return seg_loss

def calculate_total_flux(flux_data, V_epi, V_hypo, t):
    # print("flux shape:", flux_data.shape)
    flux = flux_data.squeeze(0)
    fnep = flux[:, 0]  # simulated net ecosystem production flux
    fmineral = flux[:, 1]  # simulated mineralisation flux( hypo NEP)
    fsed = flux[:, 2]  # simulated net sedimentation flux
    fatm = flux[:, 3]  # simulated atmospheric exchange flux
    fdiff = flux[:, 4]  # simulated diffusion flux
    fentr_epi = flux[:, 5]  # simulated entrainment flux (epilimnion)
    fentr_hyp = flux[:, 6]  # simulated entrainment flux (hypolimnion)

    mixed_mask = V_hypo == 0

    V_total = V_epi[0] + V_hypo[0]
    Flux_total = torch.zeros_like(fatm)

    # mixed
    Flux_total[mixed_mask] = fatm[mixed_mask] + fnep[mixed_mask] + fmineral[mixed_mask] - fsed[mixed_mask]


    Flux_total = Flux_total * 0.001
    return Flux_total # [350]

def calculate_stratified_DOC_conservation_loss(flux_data, pred, doc_threshold, t, use_gpu):
    debug = False
    Volumes = flux_data[:,:,:2] # size: [2,350,2]
    
    V_epi = Volumes[0,:,0]  # size: [350]
    V_hypo = Volumes[1,:,1] # size: [350]
    if torch.all(V_hypo == 0):
        return torch.tensor(0.0), torch.tensor(0.0)

    V_both = torch.cat((V_epi.unsqueeze(0), V_hypo.unsqueeze(0)), dim=0) # size: [2,350]
    
    stratified_segs =  find_stratified_segments(V_hypo)
    Flux_Data = flux_data[0,:,2:9]
    
    Pred = pred.squeeze(2) # size: [2, 350]
    time_threshold = 0

    stratified_DO_loss = torch.tensor([])
    for segment in stratified_segs:
        if segment[1] - segment[0] >= time_threshold:
            segment_loss = calculate_loss_for_segment(segment, V_both, Flux_Data, Pred, t, use_gpu)
            if stratified_DO_loss.numel() == 0:
                stratified_DO_loss = segment_loss
            else:
                stratified_DO_loss = torch.cat((stratified_DO_loss, segment_loss), dim=1)

    stratified_DO_loss = stratified_DO_loss.abs()
    # stratified_loss = torch.clamp(stratified_DO_loss - 0, min=0)
    mae_stratified_loss_upper = stratified_DO_loss[::2,:].mean()
    mae_stratified_loss_lower = stratified_DO_loss[1::2,:].mean()

    return mae_stratified_loss_upper, mae_stratified_loss_lower

def calculate_total_DOC_conservation_loss(flux_data, pred, doc_threshold, t, use_gpu):
    debug = False
    Volumes = flux_data[:,:,:2] # size: [2,350,2]
    V_epi = Volumes[0,:,0]  # size: [350]
    V_hypo = Volumes[1,:,1] # size: [350]
    # V_both = torch.cat((V_epi.unsqueeze(0), V_hypo.unsqueeze(0)), dim=0) # size: [2,350]
    V_total = V_epi[0] + V_hypo[0] # For a lake, the volume is a constant.
    Mixed_mask = V_hypo == 0
    Mixed_mask = Mixed_mask[1:].unsqueeze(0)
    if debug:
        V_hypo = Volumes[1,:,1]
        
    Flux_Data = flux_data[0,:,2:9]
    Pred = pred.squeeze(2) # size: [2, 350]

    assert not torch.isnan(V_total).any(), "loss has nan V_total!!!"
    assert not torch.isnan(Flux_Data).any(), "loss has nan Flux_Data!!!"

    Flux = calculate_total_flux(Flux_Data, V_epi, V_hypo, t)

    Do_pre = Pred[::2,:-1]
    Do_next = Pred[::2,1:] 

    Do_pred_flux = Do_next - Do_pre 
    total_DO_loss = Do_pred_flux - Flux[1:]
    if debug:
        print("Do_pred_flux", Do_pred_flux)
        print("Flux", Flux[1:])
    total_DO_loss = torch.clamp(total_DO_loss - doc_threshold, min=0) #
    assert not torch.isnan(total_DO_loss).any(), "total_DO_loss contains NaN values"
    assert not torch.isnan(Mixed_mask).any(), "Mixed_mask contains NaN values"

    # mae_tota_loss = torch.tensor(0.0, device='cuda' if use_gpu else 'cpu')
    mae_tota_loss = torch.tensor(0.0)
    if Mixed_mask.any():
        # Calculate mae_tota_loss
        mae_tota_loss = torch.mean(torch.abs(total_DO_loss[Mixed_mask]))
        # print("Warning: Mixed_mask has no True values. Setting mae_tota_loss to 0.")
    assert not torch.isnan(mae_tota_loss).any(), "mae_tota_loss contains NaN values"

    return mae_tota_loss

def transformTempToDensity(temp, use_gpu):
    # print(temp)
    #converts temperature to density
    #parameter:
        #@temp: single value or array of temperatures to be transformed
    densities = torch.empty_like(temp)
    if use_gpu:
        temp = temp.cuda()
        densities = densities.cuda()
    # return densities
    # print(densities.size()
    # print(temp.size())
    densities[:] = 1000.0*(1.0-((temp[:]+288.9414)*torch.pow(temp[:] - 3.9863, 2.0))/(508929.2*(temp[:]+68.12963)))
    # densities[:] = 1000*(1-((temp[:]+288.9414)*torch.pow(temp[:] - 3.9863))/(508929.2*(temp[:]+68.12963)))
    # print("DENSITIES")
    # for i in range(10):
    #     print(densities[i,i])

    return densities

#Iterator through multiple dataloaders
class MyIter(object):
  """An iterator."""
  def __init__(self, my_loader):
    self.my_loader = my_loader
    self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]
    # print("init", self.loader_iters)

  def __iter__(self):
    return self

  def __next__(self):
    # When the shortest loader (the one with minimum number of batches)
    # terminates, this iterator will terminates.
    # The `StopIteration` raised inside that shortest loader's `__next__`
    # method will in turn gets out of this `__next__` method.
    # print("next",     print(self.loader_iters))
    
    ### raw
    # batches = [loader_iter.next() for loader_iter in self.loader_iters]

    batches = [next(loader_iter) for loader_iter in self.loader_iters]

    return self.my_loader.combine_batch(batches)

  # Python 2 compatibility
  next = __next__

  def __len__(self):
    return len(self.my_loader)

#wrapper class for multiple dataloaders
class MultiLoader(object):
  """This class wraps several pytorch DataLoader objects, allowing each time 
  taking a batch from each of them and then combining these several batches 
  into one. This class mimics the `for batch in loader:` interface of 
  pytorch `DataLoader`.
  Args: 
    loaders: a list or tuple of pytorch DataLoader objects
  """
  def __init__(self, loaders):
    self.loaders = loaders

  def __iter__(self):
    return MyIter(self)

  def __len__(self):
    l =  min([len(loader) for loader in self.loaders])
    # print(l)
    return l

  # Customize the behavior of combining batches here.
  def combine_batch(self, batches):
    return batches
  

def clip_grad_norm_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm
