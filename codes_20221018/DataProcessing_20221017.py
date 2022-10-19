# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 11:48:10 2022

@author: e0306210
"""

from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import datetime
from dateutil.relativedelta import relativedelta
from os.path import exists


# =============================================================================
#
#                       Parameters Configurations
#
# =============================================================================

label_threshold = 0
sequence_len = 20
month_gap = 2           # validation and test period gap, unit: month
months_length = 62      # 5 years, unit: month

dimensions = ['index','personid', 'TRANDATE', 'month_order', 'shares', 'price', 
   'year', 'trancode', 'gap', 'profit_rdq_future', ### add three variables
   'total_vol', 'ceo', 'cfo', 'chairman', 'director', 'officers',
   'shareholders', 'insidertenure', 'FirmSize', 'Dummy_FirmSize',
   'BidAskSpread', 'Dummy_BidAskSpread', 'RDExpenditures',
   'Dummy_RDExpenditures', 'EarningQuality', 'Dummy_EarningQuality',
   'BookTaxDifference', 'Dummy_BookTaxDifference', 'Dummy_IB',
   'BookToMarketRatio', 'Dummy_BookToMarketRatio', 'BusinessSegments',
   'GeographicalSegments', 'InsiderDayCount', 'AnalystCoverage',
   'InstitutionalOwnership', 'Dummy_InstitutionalOwnership',
   'ShortSellerPosition', 'TradeTimingEarning', 'Dummy_TradeTimingEarning',
   'TradeTimingGuidance', 'Dummy_TradeTimingGuidance',
   'RecentAbnormalStockReturns', 'Dummy_RecentAbnormalStockReturns',
   'RecentEarningsInformation', 'Dummy_RecentEarningsInformation',
   'QualityOfInternalControl', 'Dummy_QualityOfInternalControl',
   'IndustryMembership', 'sich', 'dummy_sich', 'GrowthInSales',
   'Dummy_GrowthInSales', 'StockReturnSkewness',
   'Dummy_StockReturnSkewness', 'StockReturnVolatility',
   'Dummy_StockReturnVolatility', 'StockTurnover', 'Dummy_StockTurnover',
   'Restatement', 'Litigation', 'Misstatement', 'adj_t', 'adj_q5Return']
		# label = ['adj_t','adj_q5Return']
        
         
# =============================================================================
#
#                       Self-defined Function
#
# =============================================================================

 
def stats_summary(df):
    summary = df.describe() 
    (rows, cols) = df.shape 
    
    vars_list = df.columns
    
    dash = '-' * 160 
    print(dash+'\n') 
    print('{:<35s}{:>10s}{:>10s}{:>15s}{:>15s}{:>15s}{:>15s}{:>15s}{:>15s}{:>15s}\n'.format('Column',\
                                                        'Count', 'Missing', 'Mean','Std','Min','25%','50%','75%','Max')) 
    print(dash+'\n') 

    for n_cols in range(len(vars_list)):
        #col_name = df.columns[n_cols]
        col_name = vars_list[n_cols]
        col_count = summary[col_name]['count']
        col_missing = rows - summary[col_name]['count']
        col_mean  = summary[col_name]['mean']

        col_std = summary[col_name]['std']
        col_min = summary[col_name]['min']
        col_25perc = summary[col_name]['25%']
        col_50perc = summary[col_name]['50%']
        col_75perc = summary[col_name]['75%']
        col_max = summary[col_name]['max']

        print('{:<35s}{:>10.0f}{:>10.0f}{:>15f}{:>15f}{:>15f}{:>15f}{:>15f}{:>15f}{:>15f}\n'.format(str(col_name), \
                        col_count, col_missing, col_mean, col_std, col_min, col_25perc,col_50perc, col_75perc,col_max  ))

    print('\n')
    
    
        
# =============================================================================
#
#                       Data Pre-processing
#
# =============================================================================


df = pd.read_csv('20221013_all_insider_trades.csv')


df['index'] = df.index
df['year'] = df['TRANDATE'].map(lambda x: int(x.split('/')[0]))
		
		
def gen_month(x):
    year = x.split('/')[0]
    month = x.split('/')[1]			
    ans = (int(year)-2003)*12+int(month)
    return ans

df['month_order'] = df['TRANDATE'].map(gen_month)
		
		
#self.df['sich_1'] = self.df['sich'] #//100
#dummy_sich = pd.get_dummies(self.df['sich_1'])
#sic2 = list(dummy_sich.columns)
#sic2.remove(0.0)
#self.df = pd.concat([self.df, dummy_sich], axis = 1)
#print(self.df.columns)
		
#dimensions.extend(sic2)
		
		
MyData = pd.DataFrame(data=df, columns=dimensions)
#MyData['year'] = MyData['TRANDATE'].map(lambda x: int(x.split('/')[0]))
MyData['TRANDATE'] = MyData['TRANDATE'].map(lambda x: int(x.replace('/', '')))

def threshold(x):
    if x > label_threshold:
        return 1
    else:
        return 0

MyData['label'] = MyData['adj_t'].map(threshold)
MyData = MyData.drop(['adj_t', 'adj_q5Return'], axis=1)
print(MyData.shape)

MyData = MyData.dropna()
print(MyData.shape)



# =============================================================================
#
#                       Data Normalization
#
# =============================================================================

regular = ['gap', 'profit_rdq_future', 'shares', 'price', 'total_vol', \
           'FirmSize', 'TradeTimingEarning', 'AnalystCoverage',\
           'TradeTimingGuidance', 'StockTurnover',  'BidAskSpread', 'RecentEarningsInformation']

data_ = MyData.copy()

for i in regular:
       data_[i + '_min'] = 0
       data_[i + '_den'] = 0

for monthi in data_.month_order.unique():    
    BenMonth = monthi - month_gap if monthi > month_gap else monthi    
    presample = data_[data_['month_order'] <= BenMonth]
    presample = data_[data_['month_order'] >= BenMonth - months_length]
    for reg in regular:
        data_.loc[data_['month_order']== monthi, reg +'_min'] = presample[reg].min()
        data_.loc[data_['month_order']== monthi, reg +'_den'] = presample[reg].max() - presample[reg].min()    

for reg in regular:
    data_[reg +'_norm'] = (data_[reg] - data_[reg +'_min']) 
    data_[reg +'_norm'] = data_[reg +'_norm'].div(data_[reg +'_den'].values)


MyData = data_.copy()
for reg in regular:
    MyData = MyData.drop(reg, axis=1)
    MyData = MyData.drop(reg +'_min', axis=1)
    MyData = MyData.drop(reg +'_den', axis=1)
		
		
print('standard complete!')
stats_summary(MyData)
print(MyData.shape)
assert MyData.shape[1] == 63
MyData = MyData.sort_values('TRANDATE', ascending=True)
#print(MyData.shape)



# =============================================================================
#
#                       Data Grouping
#
# =============================================================================



data = MyData.sort_values(['personid', 'TRANDATE'], ascending=[True, True])
res_df = pd.DataFrame()
grouped = data.groupby('personid')

for name, group in grouped:
    '''
        Adjust gap and profit_rdg_future    
        针对这两个变量。假定一个人有三笔交易，那么第一笔交易都为空。第二笔交易，如果距离第一笔交易的间隔超过了 month_gap 个月，
        那么第二笔交易的这两个变量用第一笔交易的两个变量来填充。如果交易间隔短语 month_gap 个月，就用0填充。
    '''
    group['gap_lag'] = group['gap_norm'].shift(1)
    group['profit_rdq_future_lag'] = group['profit_rdq_future_norm'].shift(1)            
    group['trandate_lag'] = group['TRANDATE'].shift(1)
    
    group['trandate_mago'] = group['TRANDATE'].map(lambda x:datetime.datetime(year=int(x/10000),\
                                               month=int((x%10000)/100),\
                                               day=int(((x%10000)%100)))+ relativedelta(months=-month_gap))
    
    group['trandate_mago_int'] = group['trandate_mago'].map(lambda x: (10000*x.year + 100*x.month + x.day))
    
    group.loc[group['trandate_lag'] > group['trandate_mago_int'], 'gap_lag'] = 0
    group.loc[group['trandate_lag'] > group['trandate_mago_int'], 'profit_rdq_future_lag'] = 0

    group = group.drop(['gap_norm', 'profit_rdq_future_norm', \
                        'trandate_lag', 'trandate_mago', \
                        'trandate_mago_int'], axis=1)
        
    group = group.rename(columns={
                "gap_lag":"gap_norm",
                "profit_rdq_future_lag":"profit_rdq_future_norm"})
   
    group["gap_norm"] = group["gap_norm"].fillna(0)
    group["profit_rdq_future_norm"] = group["profit_rdq_future_norm"].fillna(0)
    
    
    '''
        rolling window         
        add obs in rolling window method    
    '''
    col = group.columns.tolist()
    d_l = []
    slice_ = [0]*len(col)
    for i in range(1, sequence_len):
        d_l.append(slice_)            
    d_l += group.values.tolist()  
    
    for i in range(sequence_len, len(d_l)+1, 1):
        d = pd.DataFrame(d_l[i-sequence_len: i], columns=col)
        d['index']          = d.loc[sequence_len-1, 'index']
        d['personid']       = d.loc[sequence_len-1, 'personid']
        d['TRANDATE']       = d.loc[sequence_len-1, 'TRANDATE']
        d['month_order']    = d.loc[sequence_len-1, 'month_order']
        res_df = res_df.append(d)
        
        
# =============================================================================
#
#                       DATA Generating
#
# =============================================================================

dimensions = res_df.columns.to_list()
dimensions.remove('index')
dimensions.remove('personid')
dimensions.remove('TRANDATE')
dimensions.remove('month_order')
dimensions.remove('label')
print(dimensions)
print(len(dimensions))
assert len(dimensions) == 58

for year in range(2007, 2018):
    for month in range(1, 13):    
        test_month = (year-2003)*12 + month + month_gap      
        train_month_end = test_month - month_gap
        train_month_start = test_month - months_length     
        
        
        train_data = res_df.loc[res_df.loc[:,'month_order']<=train_month_end,:]        
        train_data = train_data.loc[train_data.loc[:,'month_order']>=train_month_start, :]
        assert train_data.shape[0]%sequence_len == 0 
        train_x = train_data[dimensions].values
        print(train_x.shape[0] / sequence_len)
        print(train_x.shape[0])
        train_x = train_x.reshape((int(train_x.shape[0]/sequence_len), sequence_len, 58))
        train_y = train_data['label'].values
        train_y = train_y.reshape((int(train_data.shape[0]/sequence_len), sequence_len))
        train_y = train_y[:, -1]
        print(f"train_dataset x's shape {train_x.shape}" )
        print(f"train_dataset y's shape {train_y.shape}" )   
        np.save(f"data/train_x_{train_month_end}.npy", train_x)
        np.save(f"data/train_y_{label_threshold}_{train_month_end}.npy", train_y)

 
        test_data = res_df.loc[res_df.loc[:,'month_order']==test_month, :]
        assert test_data.shape[0]%sequence_len == 0     
        test_x = test_data[dimensions].values
        print(test_x.shape[0] / sequence_len)
        print(test_x.shape[0])
        test_x = test_x.reshape((int(test_x.shape[0]/sequence_len), sequence_len, 58))
        test_y = test_data['label'].values
        test_y = test_y.reshape((int(test_data.shape[0]/sequence_len), sequence_len))
        test_y = test_y[:, -1]
        print(f"test_dataset x's shape {test_x.shape}" )
        print(f"test_dataset y's shape {test_y.shape}" )   
        np.save(f"data/test_x_{test_month}.npy", test_x)
        np.save(f"data/test_y_{label_threshold}_{test_month}.npy", test_y)



