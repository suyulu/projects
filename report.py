import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

from data import InsiderTrades
from money_data import Model #### ???? should be ---> from model import Model


def main(year, epoch):
    df = pd.read_csv('20220422_all_insider_trades.csv')
    report_dataset = InsiderTrades("report", df, -1)
    print(f"Report dataset length: {len(report_dataset)}")
    report_loader = DataLoader(
    dataset=report_dataset,      # 数据，封装进Data.TensorDataset()类的数据
    batch_size=1,      # 每块的大小
    shuffle=False,               # 要不要打乱数据 (打乱比较好)
    num_workers=8
    )

    device = torch.device('cuda:0')

    model = Model(55, 128, 1, 2, device)  ## how to select hidden layer dimensions 128 and hidden layers 2???
    model.to(device)
    ckpt = torch.load(f'ckpt/year{year}_epoch{epoch}.pth', map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model = model.eval()
    print("Weights loaded.")

    with open(f'report_thres128_ndcg10_year{year}.csv', 'w') as f:
        f.write('index,pred\n')

        with torch.no_grad():
            # for x, y, index in tqdm(report_loader, desc=f"year {year}"):
            for x, y, index in report_loader:
                x = x.to(device).float()
                pred = model(x)
                f.write(f'{int(index.item())},{pred[:, -1, 0].item()}\n')
    
    print("Done!")


if __name__ == "__main__":
    best = {
        2007: 4,
        2008: 6,
        2009: 3,
        2010: 8,
        2011: 5,
        2012: 8,
        2013: 6,
        2014: 9,
        2015: 2,
        2016: 4,
        2017: 5
    }
    for year in range(2007, 2018):
        print(f"{year}=======================================")
        main(year, best[year])