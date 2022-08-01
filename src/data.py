from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset

class InsiderTrades(Dataset):
    def __init__(self, split, df, year, dbx=None, dby=None):
        self.split = split
        self.df = df
        self.year = year
        self.dbx = dbx
        self.dby = dby
        self.seq_len = 100

        if self.split == 'train' and (self.dbx is None or self.dby is None):
            data = self.load_data()
            self.data = data[data['TRANDATE'] <= int(str(year) + '1231')]
            print(self.data.shape)
            self.x, self.y, _ = self.group(self.seq_len)
        elif self.split == 'test':
            data = self.load_data()
            self.data = data[data['TRANDATE'] <= int(str(year) + '1231')] ## it should be (year -2) ???
            self.cur_year = self.data[self.data['TRANDATE'] >= int(str(year) + '0101')]
        elif self.split == 'report':
            data = self.load_data()
            self.cur_year = self.data = data


    def __getitem__(self, index):  
        if self.split == "train":
            return self.get_train_item(index)
        else:
            return self.get_test_item(index)

    def get_train_item(self, index) -> np.ndarray:
        if self.dbx is not None and self.dby is not None:
            return self.dbx[index], self.dby[index]   # (len_seq, d)
        else:
            return self.x[index], self.y[index]

    def get_test_item(self, index):  
        cur_item = self.cur_year.iloc[index]
        cur_idx = cur_item['index']
        cur_date = cur_item['TRANDATE']
        personid = cur_item['personid']
        seq = self.data[self.data['personid'] == personid]
        seq = seq[seq['TRANDATE'] <= cur_date]
        feat = seq.columns.tolist()[3:]
        feat.remove('label')
        x = seq[feat].values  
        ##print("******" + str(len(x)) )
        d_l = []
       
        pad_size = self.seq_len - len(x)
        if pad_size > 0 :
            slice_ = [0]*len(feat)
            for i in range(pad_size):
                d_l.append(slice_)            
            d_l += x.tolist()                    
        else:
            d_l = x.tolist()[len(x)-self.seq_len: len(x)]
		
        x = np.array(d_l)
        assert x.shape[-1] == 55
        y = seq['label'].values[-1] 
        return x, y, cur_idx

    def __len__(self):
        if self.split == 'train':
            if self.dbx is not None:
                return self.dbx.shape[0]
            else:
                return self.x.shape[0]
        else:
            return len(self.cur_year)

    def load_data(self) -> pd.DataFrame:
        # dimensions = ['PERSONID',
        #         'TRANDATE', 'CEO', 'CFO', 'CHAIRMAN', 'DIRECTOR', 'Officers',
        #         'shareholders', 'TRANCODE', 'SHARES', 'price', 'gap',
        #         'profit_rdq_future', 'total_vol',
        #         'adj_t', 'adj_q5Return']
        dimensions = ['index','personid', 'TRANDATE', 'shares', 'price',
       'year',
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
        
        self.df['index'] = self.df.index
        self.df['year'] = self.df['TRANDATE'].map(lambda x: int(x.split('/')[0]))
        
        
        MyData = pd.DataFrame(data=self.df, columns=dimensions)
        #MyData['year'] = MyData['TRANDATE'].map(lambda x: int(x.split('/')[0]))
        MyData['TRANDATE'] = MyData['TRANDATE'].map(lambda x: int(x.replace('/', '')))

        def threshold(x):
            if x > 1.66:
                return 1
            else:
                return 0

        MyData['label'] = MyData['adj_t'].map(threshold)
        MyData = MyData.drop(['adj_t', 'adj_q5Return'], axis=1)
        print(MyData.shape)
        # assert MyData.shape == (1287036, 15)
        
        MyData = MyData.dropna()
        
        regular = ['shares', 'price', 'total_vol', 'FirmSize', 'TradeTimingEarning', 'AnalystCoverage', 'TradeTimingGuidance', 'StockTurnover',  'BidAskSpread', 'RecentEarningsInformation']
        
        
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
        assert MyData.shape[1] == 59
        MyData = MyData.sort_values('TRANDATE', ascending=True)
        #print(MyData.shape)
        print("data has been loaded =========================================================================")
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
            pad_size = seq_len - len(group)
            if pad_size > 0 :
                slice_ = [0]*len(col)
                for i in range(pad_size):
                    d_l.append(slice_)            
                d_l += group.values.tolist()                    
            else:
                for i in range(seq_len, len(group)+1):
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
        assert len(dimensions) == 55
        x = d[dimensions].values
        print(len_after_padding / seq_len)
        print(len_after_padding)
        x = x.reshape((int(len_after_padding / seq_len), seq_len, 55))
        y = d['label'].values
        y = y.reshape((int(len_after_padding / seq_len), seq_len))
        np.save("data/" + self.split + str(self.year) + "_x.npy", x)
        np.save("data/" + self.split + str(self.year) + "_y.npy", y)
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

def main():
    ##x_train = np.load("data/train_x.npy")
    ##y_train = np.load("data/train_y.npy")
    df = pd.read_csv('20220422_all_insider_trades.csv')
    
    ds = InsiderTrades('train', df, 2007)
    return ds


if __name__ == "__main__":
    main()