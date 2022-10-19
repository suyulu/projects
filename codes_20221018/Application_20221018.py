# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:50:07 2022

@author: e0306210
"""

import pandas as pd
import math
import time
import os
import numpy as np
import sklearn
from torch.utils.data import TensorDataset
import torch
import torch.nn  as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics
from LSTM import LSTMModel, FocalLoss
import argparse
import itertools



np.random.seed(1425)


label_threshold = 0
sequence_len = 20
month_gap = 2           # validation and test period gap, unit: month
months_length = 62      # 5 years, unit: month

# Number of steps to unroll
seq_dim = 20 

'''
    LSTM Hyper-Parameters

'''

input_dim = 58
hidden_dim = 128
layer_dim = 2  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
output_dim = 1

parameters = dict(
  		train_batch_size	= [2048],
  		lr 					= [0.001],
  		dropout				= [0.3],
  		num_epoch			= [40],
  		lr_sche_step		= [5000]
	)



def ndcgk_precision_sensitivity(actual, prediction, k_int):
    '''
    Parameters
    ----------
    actual : np.array
        actual .
    prediction : np.array
        prediction .
    k_int : int
        k value.

    Returns
    -------
    list
        results.
    '''
    # ndcg@k
    #ndcg_k = ndcg_score([actual], [prediction], k= k_int)
    
    sorted_prediction = np.argsort(-prediction) # here, descendingly sort
    sorted_actual = actual[sorted_prediction]
    
    # ndcg@k from Bao et al., (2019)
    n = actual[actual==1].shape[0]
    if k_int < n:
        n = k_int
    
    z = 0
    for i in range(n):
        rel = 1
        z = z+ (2^rel-1)/math.log(2+i,2)
    
    ndcg_k = 0
    for j in range(k_int):
        if sorted_actual[j] == 1:
            rel = 1
            ndcg_k = ndcg_k +  (2^rel-1)/math.log(2+j,2); 
    
    if z!=0:
        ndcg_at_k = ndcg_k/z
    else:
        ndcg_at_k = 0
        
    # Precision and Sensitivity
    
    tp = np.sum(sorted_actual[0:k_int])
    fn = np.sum(sorted_actual[k_int:])
    
    
    precision_k = tp/k_int
    sensitivity_k = tp/(tp+fn)
    
    return [k_int, ndcg_at_k,  precision_k, sensitivity_k]


def train(model,optimizer,scheduler,criterion,train_loader,device,epoch,args,clip=1):
    for step, (images, labels) in enumerate(train_loader):
        # Load images as Variable
        images = images.view(-1, seq_dim, input_dim).requires_grad_().to(device).float()
        labels = labels.to(device).float()
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
        # Forward pass to get output/logits
        # outputs.size() --> 100, 10
        outputs = model(images)
        auc_step = metrics.roc_auc_score(labels.detach().cpu().int().numpy(), outputs.detach().cpu().numpy())
        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)
        # Getting gradients w.r.t. parameters
        loss.backward()
        # Updating parameters
        optimizer.step()
        scheduler.step()
    return loss, auc_step


def test(model,optimizer,criterion,test_loader,device,year,month):
    i_output = []
    i_label = []
    for images, labels in test_loader:
        images = images.view(-1, seq_dim, input_dim).requires_grad_().to(device).float()
        labels = labels.to(device).int()
        # Forward pass only to get logits/output
        outputs = model(images) 
        i_label.extend(labels.tolist())
        i_output.extend(outputs.flatten().tolist())
    preds = [i if i>=0 and i<=1 else 0 for i in i_output]
    pd_res = pd.DataFrame({'label':i_label, 'preds':preds})
    pd_res.to_csv(f'{args.res_dir}/prediction/thres{args.threshold}_{year}-{month}_pred.csv')
    # with open(f'{args.res_dir}/prediction/thres{args.threshold}_year{year}_pred.csv', 'w') as fl:
    #      for i in range(len(preds)):
    #          fl.write(f"{preds[i]}\n")
    return metrics.roc_auc_score(i_label,preds), ndcgk_precision_sensitivity(np.array(i_label), np.array(preds), int(len(preds) * 0.1))

### Hyperparameter Tuning:https://towardsdatascience.com/a-complete-guide-to-using-tensorboard-with-pytorch-53cb2301e8c3
def paras_tuning(year, month,  args=None):
    
	test_month = (year-2003)*12 + month   
	train_month_end = test_month - month_gap
	   
	train_x = np.load(f"data/train_x_{train_month_end}.npy")
	train_y = np.load(f"data/train_y_{label_threshold}_{train_month_end}.npy")
    
	print(f"loading...data/train_x_{train_month_end}.npy" )
	print(f"loading...data/train_y_{label_threshold}_{train_month_end}.npy")  

	train_tensor_x = torch.Tensor(train_x) # transform to torch tensor
	train_tensor_y = torch.Tensor(train_y)    
	val_train_dataset = TensorDataset(train_tensor_x,train_tensor_y) 

	print(f"train_dataset x's shape {train_x.shape}" )
	print(f"train_dataset y's shape {train_y.shape}" )      
        
        
	test_x = np.load(f"data/test_x_{test_month}.npy")
	test_y = np.load(f"data/test_y_{label_threshold}_{test_month}.npy")
    
	print(f"loading...data/test_x_{test_month}.npy" )
	print(f"loading...data/test_y_{label_threshold}_{test_month}.npy")      
    
	test_tensor_x = torch.Tensor(test_x) # transform to torch tensor
	test_tensor_y = torch.Tensor(test_y)    
	val_val_dataset = TensorDataset(test_tensor_x,test_tensor_y) 	
    
	print(f"test_dataset x's shape {test_x.shape}" )
	print(f"test_dataset y's shape {test_y.shape}" )   
	
	param_keys   = [k for k in parameters.keys()]
	param_values = [v for v in parameters.values()]
	param_range  = list(itertools.product(*param_values))
	param_df 	 = pd.DataFrame(param_range,columns =param_keys)
	
	param_df['auc']		= 0
	param_df['ndcg']	= 0
	for index, row in param_df.iterrows():
	
		i_train_bat_size	= int(row['train_batch_size'])
		i_lr				= row['lr']
		i_dropout			= row['dropout']
		i_num_epoch			= int(row['num_epoch'])
		i_lr_sche_step 		= row['lr_sche_step']
        
		print(f"[Validation {test_month}] hyper-parameters: train_batch_size:{i_train_bat_size}, learning_rate:{i_lr}, dropout:{i_dropout},  num_epoch:{i_num_epoch}, lr_sche_step:{i_lr_sche_step}")
		
		val_train_loader = DataLoader(
									dataset		= val_train_dataset,      # 数据，封装进Data.TensorDataset()类的数据
									batch_size	= i_train_bat_size,      # 每块的大小
									shuffle		= False,  				 # 要不要打乱数据 (打乱比较好)
									drop_last	= True,
									num_workers	= 0
									)
		val_val_loader = DataLoader(
									dataset		= val_val_dataset,      # 数据，封装进Data.TensorDataset()类的数据
									batch_size	= 4096,   # 每块的大小
									shuffle		= False,                # 要不要打乱数据 (打乱比较好)
									drop_last	= True,
									num_workers	= 0
									)

		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
		if label_threshold == 0:
				criterion = nn.BCELoss()
		else:
				criterion = FocalLoss()
                
		model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, dropout=i_dropout, device =device)
		model = model.to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=i_lr)
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, i_lr_sche_step, 0.1)
		EPOCHS = i_num_epoch
        
		best_auc = 0
		best_ndcg = 0
		best_epoch = 0

		for epoch in range(EPOCHS):
			loss, auc = train(model,optimizer,scheduler,criterion,val_train_loader,device,epoch,args)
			print(f"[Validation {test_month}] Train Average, Epoch {epoch}, Loss = {loss}, Auc={auc}")
			
			auc, ndcg_dict = test(model,optimizer,criterion,val_val_loader,device,year)
			print(f"[Validation {test_month}] Evaluate auc = {auc}, ndcg_k = {ndcg_dict[1]}")
            
			if args.metric == 'AUC':
			    if auc > best_auc:
			        best_auc= auc
			        best_ndcg = ndcg_dict[1]
			        best_epoch = epoch
			else:
			    if ndcg_dict[1] > best_ndcg:
			        best_auc= auc
			        best_ndcg = ndcg_dict[1]
			        best_epoch = epoch

		param_df.loc[index, 'auc' ] = best_auc
		param_df.loc[index, 'ndcg'] = best_ndcg
		param_df.loc[index, 'num_epoch'] = best_epoch
	
	param_df.to_csv(f'{args.res_dir}/prediction/thres{args.threshold}_{year}{month}_params.csv')
	get_hyperparams(year, month, args)


def get_hyperparams(year, month, args=None):
    param_df = pd.read_csv(f'{args.res_dir}/prediction/thres{args.threshold}_{year}{month}_params.csv',header=0, index_col=False)
    
    if args.metric.lower() == 'auc':
        paras = param_df.loc[param_df['auc'].idxmax()].to_dict()
    if args.metric.lower() == 'ndcg':
        paras = param_df.loc[param_df['ndcg'].idxmax()].to_dict()
        
    print(f"[Train 2003-{year} to test {year+2}] best hyper-parameters: train_batch_size:{paras['train_batch_size']}, learning_rate:{paras['lr']}, dropout:{paras['dropout']}, num_epoch:{paras['num_epoch']}, lr_sche_step:{paras['lr_sche_step']}")
    return paras


def run_one_year(year, month, paras, args=None):
    
	test_month = (year-2003)*12 + month +  month_gap
	train_month_end = test_month - month_gap
        
	train_x = np.load(f"data/train_x_{train_month_end}.npy")
	train_y = np.load(f"data/train_y_{label_threshold}_{train_month_end}.npy")
    
	print(f"loading...data/train_x_{train_month_end}.npy" )
	print(f"loading...data/train_y_{label_threshold}_{train_month_end}.npy")  
            
	train_tensor_x = torch.Tensor(train_x) # transform to torch tensor
	train_tensor_y = torch.Tensor(train_y)    
	train_dataset = TensorDataset(train_tensor_x,train_tensor_y)  
    
	print(f"train_dataset x's shape {train_x.shape}" )
	print(f"train_dataset y's shape {train_y.shape}" )  
    
	test_x = np.load(f"data/test_x_{test_month}.npy")
	test_y = np.load(f"data/test_y_{label_threshold}_{test_month}.npy")
    
	print(f"loading...data/test_x_{test_month}.npy" )
	print(f"loading...data/test_y_{label_threshold}_{test_month}.npy")  
        
	test_tensor_x = torch.Tensor(test_x) # transform to torch tensor
	test_tensor_y = torch.Tensor(test_y)    
	test_dataset = TensorDataset(test_tensor_x,test_tensor_y) 
    
	print(f"test_dataset x's shape {test_x.shape}" )
	print(f"test_dataset y's shape {test_y.shape}" )   
    
	train_loader = DataLoader(
							dataset=train_dataset,      # 数据，封装进Data.TensorDataset()类的数据
							batch_size=int(paras['train_batch_size']),      # 每块的大小
							shuffle=False,  # 要不要打乱数据 (打乱比较好)
							drop_last=True,
							num_workers=0
							)
	test_loader = DataLoader(
							dataset=test_dataset,      # 数据，封装进Data.TensorDataset()类的数据
							batch_size=4096,      # 每块的大小
							shuffle=False,               # 要不要打乱数据 (打乱比较好)
							drop_last=True,
							num_workers=0
							)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	if label_threshold == 0:
		criterion = nn.BCELoss()
	else:
		criterion = FocalLoss()
        
	model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, dropout=paras['dropout'], device =device)
	model = model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=paras['lr'])
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, paras['lr_sche_step'], 0.1)
	EPOCHS = paras['num_epoch']

	for epoch in range(int(EPOCHS)):
		loss, auc = train(model,optimizer,scheduler,criterion,train_loader,device,epoch,args)
		print(f"[TEST {test_month}] Train Average Epoch {epoch}, Loss = {loss}, auc = {auc}")
        #fw.add_scalar(f'year_{year}_train_loss', loss, epoch)
        #fw.add_scalar(f'year_{year}_train_auc', auc, epoch)
		torch.save({'model': model.state_dict()}, args.res_dir + f'/checkpoints/{test_month}_epoch{epoch}.pth')
		
	auc, ndcg_dict = test(model,optimizer,criterion,test_loader,device, year)
	print(f"[TEST {test_month}] Evaluate auc = {auc}, ndcg_k = {ndcg_dict[1]}, lr = {optimizer.state_dict()['param_groups'][0]['lr']}")
	#wandb.log({"year": year, "epoch": epoch, "auc": auc, "ndcg_k": ndcg_dict[1], "lr": optimizer.state_dict()['param_groups'][0]['lr'], "loss": loss})
    #fw.add_scalar(f'year_{year}_test_auc', auc, epoch)
    #fw.add_scalar(f'year_{year}_test_ndcg', ndcg_dict[1], epoch)
        

def init_folders(args):
	#if not os.path.exists('/data'):
	#	os.makedirs('/data', exist_ok=False)
	os.makedirs(args.res_dir, exist_ok=False)
	os.makedirs(args.res_dir + '/checkpoints', exist_ok=False)
	os.makedirs(args.res_dir + '/figures', exist_ok=False)
	os.makedirs(args.res_dir + '/validation', exist_ok=False)
	os.makedirs(args.res_dir + '/prediction', exist_ok=False)

def main(args):
    init_folders(args)
    #fw = SummaryWriter(args.res_dir + '/tensorboard')
    for year in range(2007, 2018):
        for month in range(4, 13):            
            print(f"==================================Year-Month: {year}-{month}==================================")
            paras_tuning(year, month,  args)
            paras = get_hyperparams(year, month, args)
    #     year = 2010
    #     paras = dict(
    #   		train_batch_size	= 1024,
    #   		lr 					= 1e-4,
    #   		dropout				= 0.3,
    #   		num_epoch			= 40,
    #   		lr_sche_step		= 5000
    # 	)
            run_one_year(year, month, paras, args)
        

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="LSTM_project")    
    parser.add_argument("--data_path", default="20220422_all_insider_trades.csv", type=str, help="path of dataset")
    parser.add_argument("--res_dir", default="./InsiderTrading/lstm_results", type=str, help="result directory")    
    parser.add_argument("--print_freq", default=20, type=int, help="print frequency")    
    parser.add_argument("--threshold", default=0, type=float, help="threshold of adj_t")
    parser.add_argument("--metric", default='AUC', type=str, help="metrics")
	
    args = parser.parse_args()
    print(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    args.res_dir = args.res_dir + "_" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    print("args", args)

    main(args)
