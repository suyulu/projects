from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import datetime
from dateutil.relativedelta import relativedelta
from os.path import exists


data_threshold = 1.66
sequence_len = 100

class InsiderTrades(Dataset):
    def __init__(self, split, df, year, dbx=None, dby=None):
        self.split = split
        self.df = df
        self.year = year
        self.dbx = dbx
        self.dby = dby
        self.threshold = data_threshold #args.threshold
        self.seq_len = sequence_len

        if self.split == 'train' and (self.dbx is None or self.dby is None):
            if (exists("data/train" + str(year) + "_x.npy")) and (exists("data/train" + str(year) + str(self.threshold) +  "_y.npy")):
                #self.dbx = np.load("data/" + self.split + str(self.year) + "_x.npy")
                #self.dby = np.load("data/" + self.split + str(self.year) + "_y.npy")
                pass
            else:
                data = self.load_data()
                self.data = data[data['TRANDATE'] <= int(str(year) + '1231')]
                print(self.data.shape)
                self.x, self.y, _ = self.group(self.seq_len)
        elif self.split == 'test':
            if (exists("data/" + self.split + str(year) + "_x.npy")) and (exists("data/" + self.split + str(year) + str(self.threshold) +  "_y.npy")):  
                #self.dbx = np.load("data/" + self.split + str(self.year) + "_x.npy")
                #self.dby = np.load("data/" + self.split + str(self.year) + "_y.npy")
                pass
            else:                
                data = self.load_data()
                self.data = data[data['TRANDATE'] <= int(str(year) + '1231')] ## it should be (year -2) ???
                self.cur_year = self.data[self.data['TRANDATE'] >= int(str(year) + '0101')]
                self.x, self.y = self.get_test_item()
        elif self.split == 'report':
            data = self.load_data()
            self.cur_year = self.data = data


    def __getitem__(self, index):  
        return self.get_train_item(index)


    def get_train_item(self, index) -> np.ndarray:
        if self.dbx is not None and self.dby is not None:
            return self.dbx[index], self.dby[index]   # (len_seq, d)
        else:
            return self.x[index], self.y[index]

    def get_test_item(self):  
        l_x = []
        l_y = []
        for index, cur_item in self.cur_year.iterrows():
            cur_date = cur_item['TRANDATE']
            personid = cur_item['personid']
            seq = self.data[self.data['personid'] == personid]
            seq = seq[seq['TRANDATE'] <= cur_date]            
        
            '''
            Adjust gap and profit_rdg_future
            
            针对这两个变量。假定一个人有三笔交易，那么第一笔交易都为空。第二笔交易，如果距离第一笔交易的间隔超过了6个月，
            那么第二笔交易的这两个变量用第一笔交易的两个变量来填充。如果交易间隔短语6个月，就用0填充。
            
            '''
            seq['gap_lag'] = seq['gap_norm'].shift(1)
            seq['profit_rdq_future_lag'] = seq['profit_rdq_future_norm'].shift(1)        
            seq['trandate_lag'] = seq['TRANDATE'].shift(1)
            
            seq['trandate_6mago'] = seq['TRANDATE'].map(lambda x:datetime.datetime(year=int(x/10000),\
                                                       month=int((x%10000)/100),\
                                                       day=int(((x%10000)%100)))+ relativedelta(months=-6))
            
            seq['trandate_6mago_int'] = seq['trandate_6mago'].map(lambda x: (10000*x.year + 100*x.month + x.day))
            
            seq.loc[seq['trandate_lag'] > seq['trandate_6mago_int'], 'gap_lag'] = 0
            seq.loc[seq['trandate_lag'] > seq['trandate_6mago_int'], 'profit_rdq_future_lag'] = 0
    
            seq = seq.drop(['gap_norm', 'profit_rdq_future_norm', \
                                'trandate_lag', 'trandate_6mago', \
                                'trandate_6mago_int'], axis=1)
                
            seq = seq.rename(columns={
                        "gap_lag":"gap_norm",
                        "profit_rdq_future_lag":"profit_rdq_future_norm"})
           
            seq["gap_norm"] = seq["gap_norm"].fillna(0)
            seq["profit_rdq_future_norm"] = seq["profit_rdq_future_norm"].fillna(0)
            
                
            feat = seq.columns.tolist()[3:]
            feat.remove('label')
            x = seq[feat].values  
            ##print("******" + str(len(x)) )            
            d_x = []
            l_y.append(seq['label'].values.tolist()[-1])
            pad_size = self.seq_len - len(x)
            if pad_size > 0 :
                slice_ = [0]*len(feat)
                for i in range(pad_size):
                    d_x.append(slice_)            
                d_x += x.tolist()                    
            else:
                d_x = x.tolist()[len(x)-self.seq_len: len(x)]
            
            
            l_x += d_x
            
        x = np.array(l_x)
        y = np.array(l_y)
        assert x.shape[-1] == 128
        x = x.reshape(-1, self.seq_len, 128)
        y = y.reshape(x.shape[0], 1)
        print(f"test_dataset x's shape {x.shape}" )
        print(f"test_dataset y's shape {y.shape}" )
        if not (exists("data/" + self.split + str(self.year) + "_x.npy")):
            np.save("data/" + self.split + str(self.year) + "_x.npy", x)
        np.save("data/" + self.split + str(self.year) + "_" + str(self.threshold) + "_y.npy", y)
        return x, y

    def __len__(self):
        if self.dbx is not None:
            return self.dbx.shape[0]
        else:
            return self.x.shape[0]

    def load_data(self) -> pd.DataFrame:
        # dimensions = ['PERSONID',
        #         'TRANDATE', 'CEO', 'CFO', 'CHAIRMAN', 'DIRECTOR', 'Officers',
        #         'shareholders', 'TRANCODE', 'SHARES', 'price', 'gap',
        #         'profit_rdq_future', 'total_vol',
        #         'adj_t', 'adj_q5Return']
        dimensions = ['index','personid', 'TRANDATE', 'shares', 'price',
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
       'IndustryMembership', 'dummy_sich', 'GrowthInSales',
       'Dummy_GrowthInSales', 'StockReturnSkewness',
       'Dummy_StockReturnSkewness', 'StockReturnVolatility',
       'Dummy_StockReturnVolatility', 'StockTurnover', 'Dummy_StockTurnover',
       'Restatement', 'Litigation', 'Misstatement', 'adj_t', 'adj_q5Return']
		# label = ['adj_t','adj_q5Return']
        
        self.df['index'] = self.df.index
        self.df['year'] = self.df['TRANDATE'].map(lambda x: int(x.split('/')[0]))
        
        self.df['sich_1'] = self.df['sich'] #//100
        dummy_sich = pd.get_dummies(self.df['sich_1'])
        sic2 = list(dummy_sich.columns)
        sic2.remove(0.0)
        self.df = pd.concat([self.df, dummy_sich], axis = 1)
        print(self.df.columns)
        
		
        dimensions.extend(sic2)
		
		
        MyData = pd.DataFrame(data=self.df, columns=dimensions)
        #MyData['year'] = MyData['TRANDATE'].map(lambda x: int(x.split('/')[0]))
        MyData['TRANDATE'] = MyData['TRANDATE'].map(lambda x: int(x.replace('/', '')))

        def threshold(x):
            if x > self.threshold:
                return 1
            else:
                return 0

        MyData['label'] = MyData['adj_t'].map(threshold)
        MyData = MyData.drop(['adj_t', 'adj_q5Return'], axis=1)
        print(MyData.shape)
        # assert MyData.shape == (1287036, 15)
        
        MyData = MyData.dropna()
        
        regular = ['gap', 'profit_rdq_future', 'shares', 'price', 'total_vol', \
                   'FirmSize', 'TradeTimingEarning', 'AnalystCoverage',\
                   'TradeTimingGuidance', 'StockTurnover',  'BidAskSpread', 'RecentEarningsInformation']
        
        
        data_ = MyData.copy()

        for i in regular:
               data_[i + '_min'] = 0
               data_[i + '_den'] = 0
        
        for yeari in data_.year.unique():           
            
            benyear = yeari-2 if yeari>2004 else yeari
            
            presample = data_[data_['year'] == benyear]

            #print(str(yeari) + " " + str(benyear))
            for reg in regular:
                data_.loc[data_['year']== yeari, reg +'_min'] = presample[reg].min()
                data_.loc[data_['year']== yeari, reg +'_den'] = presample[reg].max() - presample[reg].min()
            
        
        for reg in regular:
            data_[reg +'_norm'] = (data_[reg] - data_[reg +'_min']) 
            data_[reg +'_norm'] = data_[reg +'_norm'].div(data_[reg +'_den'].values)

        
        MyData = data_.copy()
        for reg in regular:
            MyData = MyData.drop(reg, axis=1)
            MyData = MyData.drop(reg +'_min', axis=1)
            MyData = MyData.drop(reg +'_den', axis=1)
		
		
        print('standard complete!{}'.format(MyData[reg +'_norm'].mean()))
        self.stats_summary(MyData)
        print(MyData.shape)
        assert MyData.shape[1] == 132
        MyData = MyData.sort_values('TRANDATE', ascending=True)
        #print(MyData.shape)
        print("================================= data has been loaded =================================")
        return MyData
	
	
	### test2 in test_20220720_yulu.py
    def group(self, seq_len: int) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        '''
        :param data:
        :param seq_len:
        :param mode:
        :return: numpy.ndarray, x,y
        '''
		
        self.data = self.data.sort_values(['personid', 'TRANDATE'], ascending=[True, True])
        d_l = []
        col = self.data.columns.tolist()
        grouped = self.data.groupby('personid')

        for name, group in grouped:
            '''
            Adjust gap and profit_rdg_future
            
            针对这两个变量。假定一个人有三笔交易，那么第一笔交易都为空。第二笔交易，如果距离第一笔交易的间隔超过了6个月，
            那么第二笔交易的这两个变量用第一笔交易的两个变量来填充。如果交易间隔短语6个月，就用0填充。
            
            '''
            group['gap_lag'] = group['gap_norm'].shift(1)
            group['profit_rdq_future_lag'] = group['profit_rdq_future_norm'].shift(1)            
            group['trandate_lag'] = group['TRANDATE'].shift(1)
            
            group['trandate_6mago'] = group['TRANDATE'].map(lambda x:datetime.datetime(year=int(x/10000),\
                                                       month=int((x%10000)/100),\
                                                       day=int(((x%10000)%100)))+ relativedelta(months=-6))
            
            group['trandate_6mago_int'] = group['trandate_6mago'].map(lambda x: (10000*x.year + 100*x.month + x.day))
            
            group.loc[group['trandate_lag'] > group['trandate_6mago_int'], 'gap_lag'] = 0
            group.loc[group['trandate_lag'] > group['trandate_6mago_int'], 'profit_rdq_future_lag'] = 0

            group = group.drop(['gap_norm', 'profit_rdq_future_norm', \
                                'trandate_lag', 'trandate_6mago', \
                                'trandate_6mago_int'], axis=1)
                
            group = group.rename(columns={
                        "gap_lag":"gap_norm",
                        "profit_rdq_future_lag":"profit_rdq_future_norm"})
           
            group["gap_norm"] = group["gap_norm"].fillna(0)
            group["profit_rdq_future_norm"] = group["profit_rdq_future_norm"].fillna(0)
            
            
            '''
                rolling window 
                
                add obs in rolling window method
            
            '''
            
            
            pad_size = seq_len - len(group)
            if pad_size > 0 :
                
                slice_ = [0]*len(col)
                for i in range(pad_size):
                    d_l.append(slice_)            
                d_l += group.values.tolist()                    
            else:
                for i in range(seq_len, len(group)+1, 1): # we can change the step to speed up
                    ##print("******" + str(i-seq_len) + "*********" +str(i) + "*********" + str(len(group)+1))
                    d_l += group.values.tolist()[i-seq_len: i]
		
        len_after_padding = len(d_l)

        assert len_after_padding % seq_len == 0
        '''
        col = ['PERSONID','TRANDATE', 
            'CEO', 'CFO', 'CHAIRMAN', 'DIRECTOR', 'Officers',
            'shareholders', 'TRANCODE', 'SHARES', 'price', 'gap',
            'profit_rdq_future',  'total_vol', 'label']
        '''
        d = pd.DataFrame(d_l, columns=col)
        
        # 实现效果先按照PERSONID排序，后按照TRANDATE排序

        dimensions = col[3:]
        dimensions.remove('label')
        print(dimensions)
        print(len(dimensions)) # the dimensions has "year" variable
        assert len(dimensions) == 128
        
        x = d[dimensions].values
        print(len_after_padding / seq_len)
        print(len_after_padding)
        x = x.reshape((int(len_after_padding / seq_len), seq_len, 128))
        y = d['label'].values
        y = y.reshape((int(len_after_padding / seq_len), seq_len))
        y = y[:, -1]
        print(f"train_dataset x's shape {x.shape}" )
        print(f"train_dataset y's shape {y.shape}" )
        if not (exists("data/" + self.split + str(self.year) + "_x.npy")):
            np.save("data/" + self.split + str(self.year) + "_x.npy", x)
        np.save("data/" + self.split + str(self.year) + "_" + str(self.threshold) + "_y.npy", y)
        return x, y, d
 
    
    def stats_summary(self, df):
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
        
        
        
    def to_integer(dt_time):
        return 10000*dt_time.year + 100*dt_time.month + dt_time.day


def main():
    ##x_train = np.load("data/train_x.npy")
    ##y_train = np.load("data/train_y.npy")
    
    df = pd.read_csv('20220422_all_insider_trades.csv')
    for year in range(2003, 2019):
        if (not exists("data/train" + str(year) + "_x.npy")) or (not exists("data/train" + str(year) + "_y.npy")):
            InsiderTrades('train', df, year)
        if (not exists("data/test" + str(year) + "_x.npy")) or (not exists("data/test" + str(year) + "_y.npy")):  
            InsiderTrades('test', df, year)



if __name__ == "__main__":
    main()