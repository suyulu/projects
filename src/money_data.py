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


def load_data(path):
    data = pd.read_csv(path, encoding='utf-8')

    dimensions = ['PERSONID',
                  'TRANDATE', 'CEO', 'CFO', 'CHAIRMAN', 'DIRECTOR', 'Officers',
                  'shareholders', 'TRANCODE', 'SHARES', 'price', 'gap',
                  'profit_rdq_future', 'total_vol',
                  'adj_t', 'adj_q5Return']
    # label = ['adj_t','adj_q5Return']

    MyData = pd.DataFrame(data=data, columns=dimensions)
    MyData['TRANDATE'] = MyData['TRANDATE'].map(lambda x: int(x.replace('/', '')))

    def threshold(x):
        if x > 1.6:
            return 1
        else:
            return 0

    MyData['label'] = MyData['adj_t'].map(threshold)
    MyData = MyData.drop(['adj_t', 'adj_q5Return'], axis=1)
    assert MyData.shape == (1287036, 15)
    MyData = MyData.dropna()
    #regular = ['SHARES', 'price', 'gap', 'profit_rdq_future', 'total_vol']

    # max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    #std_scaler = lambda x: (x - np.mean(x)) / np.std(x)
    #data_ = MyData.copy()
    #for i in regular:
    #    MyData[i + '_norm'] = data_[[i]].apply(std_scaler)
    #    # print('?')
    #print('standard complete!')
    #MyData = MyData.drop(regular, axis=1)
    assert MyData.shape[1]== 15
    MyData = MyData.sort_values('TRANDATE', ascending=True)
    # MyData.to_csv('./final_data.csv')
    return MyData


def group_p(data, seq_len, mode="train"):
    '''
    :param data:
    :param seq_len:
    :param mode:
    :return: numpy.ndarray, x,y
    '''
    d_l = data.values.tolist()
    col = data.columns.tolist()
    grouped = data.groupby('PERSONID')

    for name, group in grouped:
        pad_size = seq_len - len(group) % seq_len
        if pad_size != seq_len:
            slice_ = group.values.tolist()[-1]
            for i in range(pad_size):
                d_l.append(slice_)

    len_after_padding = len(d_l)

    assert len_after_padding % seq_len == 0
    '''
    col = ['PERSONID','TRANDATE', 
           'CEO', 'CFO', 'CHAIRMAN', 'DIRECTOR', 'Officers',
           'shareholders', 'TRANCODE', 'SHARES', 'price', 'gap',
           'profit_rdq_future',  'total_vol','label']
    '''
    d = pd.DataFrame(d_l, columns=col)
    d = d.sort_values(['PERSONID', 'TRANDATE'], ascending=[True, True])
    # 实现效果先按照PERSONID排序，后按照TRANDATE排序

    dimensions = col[2:]
    dimensions.remove('label')
    assert len(dimensions) == 12
    x = d[dimensions].values
    print(len_after_padding / seq_len)
    print(len_after_padding)
    x = x.reshape((int(len_after_padding / seq_len), seq_len, 12))
    y = d['label'].values
    y = y.reshape((int(len_after_padding / seq_len), seq_len))
    np.save("data/" + mode + "_x.npy", x)
    np.save("data/" + mode + "_y.npy", y)
    return x, y



def train_test(data):
    train = data[data['TRANDATE'] < 20190101]
    test = data[data['TRANDATE'] >= 20190101]
    return train, test



def prepare_data():
    path = "20220422_all_insider_trades.csv"
    data = load_data(path)
    train, test = train_test(data)
    x_train, y_train = group_p(train, 100)
    x_test, y_test = group_p(test, 100,"test")


from torch.utils.data import TensorDataset
import torch
import torch.nn  as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics
from data import InsiderTrades
from tqdm import tqdm

from model import Model, FocalLoss, GRUNet, LSTMNet

def train(model,optimizer,criterion,data_loader,device,clip=1):
    model.train()
    loss_total = 0
    auc = []
    # for x,y in tqdm(data_loader, desc='Train'):
    for x,y in data_loader:
        optimizer.zero_grad()
        x = x.to(device).float()
        y = y.to(device).float()
        #--------old
        # y = y[:,-1]
        # new
        y = y.reshape(-1)   # (100 * b,)
        pred = model(x) # (b, 100, 1)
        pred = pred.squeeze(-1).reshape(-1)
        #pred [b*seq]
        # try:
        loss = criterion(pred,y)
        # except:
        #     continue
        loss_total += loss.item()
        loss.backward()
        auc.append(metrics.roc_auc_score(y.detach().cpu().int().numpy(), pred.detach().cpu().numpy()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
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


def evaluate(model,optimizer,criterion,data_loader,device,epoch):
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
    # with open(f'res{epoch}.csv', 'w') as f:
    #     f.write("label,pred\n")
    #     for i in range(len(labels)):
    #         f.write(f"{labels[i]},{preds[i]}\n")

    return metrics.roc_auc_score(labels,preds), ndcgk_precision_sensitivity(np.array(labels), np.array(preds), int(len(preds) * 0.1))


def run_one_year(year, fw: SummaryWriter):
    df = pd.read_csv('20220422_all_insider_trades.csv')
    train_dataset = InsiderTrades("train", df, year)
    test_dataset = InsiderTrades("test", df, year+2)
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    batch_size = 256
    train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=4
    )
    test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=False,
    drop_last=True,
    num_workers=8
    )

    device = torch.device("cuda:0")

    # criterion = nn.BCELoss()
    criterion = FocalLoss()
    model = GRUNet(58, 128, 1, 2, device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    EPOCHS = 20
    for epoch in range(EPOCHS):
        loss, auc = train(model,optimizer,criterion,train_loader,device)
        print(f"in epoch {epoch}, loss = {loss}")
        fw.add_scalar(f'year_{year}_train_loss', loss, epoch)
        fw.add_scalar(f'year_{year}_train_auc', auc, epoch)
        if (epoch + 1) % 1 == 0:  ## 可以删掉吗？
            auc, ndcg_dict = evaluate(model,optimizer,criterion,test_loader,device, epoch)
            print(f"auc = {auc}, ndcg_k = {ndcg_dict[1]}")
            fw.add_scalar(f'year_{year}_test_auc', auc, epoch)
            fw.add_scalar(f'year_{year}_test_ndcg', ndcg_dict[1], epoch)
        torch.save({'model': model.state_dict()}, f'ckpt/year{year}_epoch{epoch}.pth')



def main():
    # x_train = np.load("data/train_x.npy")
    # y_train = np.load("data/train_y.npy")
    # x_test = np.load("data/test_x.npy")
    # y_test = np.load("data/test_y.npy")

    # assert np.isnan(x_train).any() == False

    # train_dataset = TensorDataset(torch.from_numpy(x_train),torch.from_numpy(y_train))
    # test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
    fw = SummaryWriter('thres166_ndcg10/tensorboard')
    for year in range(2007, 2018):
        print(f"Year: {year}====================================================================================================")
        run_one_year(year, fw)

if __name__=="__main__":
    #prepare_data()
    main()