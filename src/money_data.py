# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 18:40:47 2022

@author: 18142
"""
import pandas as pd
import math
import time
import os
import numpy as np
import sklearn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset
import torch
import torch.nn  as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics
from data import InsiderTrades
from model import Model, FocalLoss
import argparse
import numpy as np
import matplotlib.pyplot as plt
import itertools


np.random.seed(1425)


def train(model,optimizer,scheduler,criterion,data_loader,device,epoch,args,clip=1):
    model.train()
    loss_total = 0
    auc = []
    step_total = len(data_loader)
    # for x,y in tqdm(data_loader, desc='Train'):
    for step, (x, y) in enumerate(data_loader):
        optimizer.zero_grad()
        x = x.to(device).float()
        y = y.to(device).float()
        y = y.reshape(-1)   # (100 * b,)
        pred = model(x) # (b, 100, 1)
        pred = pred.squeeze(-1).reshape(-1) # (b*100*1)
        
        loss = criterion(pred,y)
        loss_total += loss.item()
        loss.backward()
        auc_step = metrics.roc_auc_score(y.detach().cpu().int().numpy(), pred.detach().cpu().numpy())
        auc.append(auc_step)

        if step % args.print_freq == 0:
            #plt.hist(list(pred.cpu().detach().numpy()))
            #plt.savefig(f"{args.res_dir}/figures/pred_distribution_epoch{epoch}_step{step}.png")
            

            statistics_dict = {}
            for key in y:
                key = int(key)
                statistics_dict[key] = statistics_dict.get(key, 0) + 1
            # print("label statistics: ", statistics_dict)
            print(f"[Train] Epoch {epoch} Step {step}/{step_total} Loss {loss} Auc {auc_step}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step()
    return loss_total, np.mean(auc)


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


def evaluate(model,optimizer,criterion,data_loader,device,year):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        # for x,y in tqdm(data_loader, desc='Evaluation'): # (1, l, 12) (1,)
        for x,y,_ in data_loader: # (1, l, 12) (1,)
            x = x.to(device).float()
            y = y.to(device).int()
            pred = model(x)

            labels.extend(y.tolist())
            # preds.extend(pred[:,-1].tolist())
            preds.extend(pred[:,-1,0].tolist())
    preds = [i if i>=0 and i<=1 else 0 for i in preds]
    with open(f'{args.res_dir}/prediction/thres{args.threshold}_year{year}_pred.csv', 'w') as fl:
         for i in range(len(preds)):
             fl.write(f"{preds[i]}\n")

    return metrics.roc_auc_score(labels,preds), ndcgk_precision_sensitivity(np.array(labels), np.array(preds), int(len(preds) * 0.1))

### Hyperparameter Tuning:https://towardsdatascience.com/a-complete-guide-to-using-tensorboard-with-pytorch-53cb2301e8c3
def one_fold_validate(year, metric = 'AUC', args=None):
	df = pd.read_csv(args.data_path)	
	val_train_dataset = InsiderTrades("train", df, year-4, args=args)
	val_val_dataset = InsiderTrades("test", df, year-2, args=args)
	
	print(f"[Validation] Training Year 2003-{year-4}, dataset length: {len(val_train_dataset)}")
	print(f"[Validation] Validation Year {year-2}, dataset length: {len(val_val_dataset)}")
	
	parameters = dict(
		train_batch_size	= [2024, 2048, 4096],
		lr 					= [0.1, 0.01, 0.001, 0.0001, 0.00001],
		dropout				= [0, 0.1, 0.3, 0.5],
		test_batch_size		= [1],
		num_epoch			= [40],
		lr_sche_step		= [5000]
	)
	
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
		i_test_batch_size	= int(row['test_batch_size'])
		i_num_epoch			= int(row['num_epoch'])
		i_lr_sche_step 		= row['lr_sche_step']
		
		val_train_loader = DataLoader(
									dataset		= val_train_dataset,      # 数据，封装进Data.TensorDataset()类的数据
									batch_size	= i_train_bat_size,      # 每块的大小
									shuffle		= False,  				 # 要不要打乱数据 (打乱比较好)
									drop_last	= True,
									num_workers	= 4
									)
		val_val_loader = DataLoader(
									dataset		= val_val_dataset,      # 数据，封装进Data.TensorDataset()类的数据
									batch_size	= i_test_batch_size,   # 每块的大小
									shuffle		= False,                # 要不要打乱数据 (打乱比较好)
									drop_last	= True,
									num_workers	= 8
									)
	
		device = torch.device("cuda:0")

		# criterion = nn.BCELoss()
		criterion = FocalLoss()
		model = Model(128, 128, 1, 2, device, dropout=i_dropout)
		model = model.to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=i_lr)
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, i_lr_sche_step, 0.1)
		EPOCHS = i_num_epoch

		for epoch in range(EPOCHS):
			loss, auc = train(model,optimizer,scheduler,criterion,val_train_loader,device,epoch,args)
			print(f"[Validation] Train Average Year {year}, Epoch {epoch}, Loss = {loss}")
			
		auc, ndcg_dict = evaluate(model,optimizer,criterion,val_val_loader,device)
		print(f"[Validation] Evaluate auc = {auc}, ndcg_k = {ndcg_dict[1]}, lr = {optimizer.state_dict()['param_groups'][0]['lr']}")
		
		param_df.loc[index, ['auc']] = auc
		param_df.loc[index, ['ndcg']] = ndcg_dict[1]
	
	if metric.lower() == 'auc':
		paras = param_df.loc[param_df['auc'].idxmax()].to_dict()
	if metric.lower() == 'ndcg':
		paras = param_df.loc[param_df['ndcg'].idxmax()].to_dict()
		
	return paras

def test(year, fw=SummaryWriter, paras=None, args=None):
    df = pd.read_csv(args.data_path)
    train_dataset = InsiderTrades("train", df, year, args=args)
    test_dataset = InsiderTrades("test", df, year+2, args=args)
    print(f"[TEST] Training dataset year 2003-{year}, length: {len(train_dataset)}")
    print(f"[TEST] Testing dataset year {year}, length: {len(test_dataset)}")

    train_loader = DataLoader(
							dataset=train_dataset,      # 数据，封装进Data.TensorDataset()类的数据
							batch_size=int(paras.train_batch_size),      # 每块的大小
							shuffle=False,  # 要不要打乱数据 (打乱比较好)
							drop_last=True,
							num_workers=4
							)
    test_loader = DataLoader(
							dataset=test_dataset,      # 数据，封装进Data.TensorDataset()类的数据
							batch_size=int(paras.test_batch_size),      # 每块的大小
							shuffle=False,               # 要不要打乱数据 (打乱比较好)
							drop_last=True,
							num_workers=8
							)
    device = torch.device("cuda:0")

    # criterion = nn.BCELoss()
    criterion = FocalLoss()
    model = Model(128, 128, 1, 2, device, dropout=paras.dropout)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=paras.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, paras.lr_sche_step, 0.1)
    EPOCHS = paras.num_epoch

    for epoch in range(int(EPOCHS)):
        loss, auc = train(model,optimizer,scheduler,criterion,train_loader,device,epoch,args)
        print(f"[TEST] Train Average Year {year}, Epoch {epoch}, Loss = {loss}")
        fw.add_scalar(f'year_{year}_train_loss', loss, epoch)
        fw.add_scalar(f'year_{year}_train_auc', auc, epoch)
        torch.save({'model': model.state_dict()}, args.res_dir + f'/checkpoints/year{year}_epoch{epoch}.pth')
		
    auc, ndcg_dict = evaluate(model,optimizer,criterion,test_loader,device, epoch, year)
    print(f"[TEST] Evaluate auc = {auc}, ndcg_k = {ndcg_dict[1]}, lr = {optimizer.state_dict()['param_groups'][0]['lr']}")
	#wandb.log({"year": year, "epoch": epoch, "auc": auc, "ndcg_k": ndcg_dict[1], "lr": optimizer.state_dict()['param_groups'][0]['lr'], "loss": loss})
    fw.add_scalar(f'year_{year}_test_auc', auc, epoch)
    fw.add_scalar(f'year_{year}_test_ndcg', ndcg_dict[1], epoch)
        

def init_folders(args):
	os.makedirs(args.res_dir, exist_ok=False)
	os.makedirs(args.res_dir + '/checkpoints', exist_ok=False)
	os.makedirs(args.res_dir + '/figures', exist_ok=False)
	os.makedirs(args.res_dir + '/validation', exist_ok=False)
	os.makedirs(args.res_dir + '/prediction', exist_ok=False)

def main(args):
    init_folders(args)
    fw = SummaryWriter(args.res_dir + '/tensorboard')
    for year in range(2007, 2018):
        print(f"Year: {year}====================================================================================================")
        paras = one_fold_validate(year,'AUC', args)
        test(year, fw, paras, args)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="LSTM_project")    
    parser.add_argument("--data_path", default="20220422_all_insider_trades.csv", type=str, help="path of dataset")
    parser.add_argument("--res_dir", default="./InsiderTrading/lstm_results", type=str, help="result directory")    
    parser.add_argument("--print_freq", default=20, type=int, help="print frequency")    
    parser.add_argument("--threshold", default=1.66, type=float, help="threshold of adj_t")
	
    args = parser.parse_args()
    print(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    args.res_dir = args.res_dir + "_" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    print("args", args)

    main(args)