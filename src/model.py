import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self,input_dim,hid_dim,output_dim,num_layers,device,dropout):
        super(Model,self).__init__()
        self.device = device
        self.rnn = nn.LSTM(input_size=input_dim,hidden_size=hid_dim,num_layers=num_layers,batch_first=True,dropout=dropout)
        # self.rnn = nn.RNN(input_size=input_dim,hidden_size=hid_dim,num_layers=num_layers,batch_first=True)
        # self.rnn = nn.GRU(input_size=input_dim,hidden_size=hid_dim,num_layers=num_layers,batch_first=True)
        self.fc = nn.Linear(hid_dim,output_dim)

    def forward(self,x):
        # x = [b,seq,f],
        assert torch.isnan(x).any() == False
        out,_ = self.rnn(x) # x is [1024, 100, 58]
        # old version-----------
        # out = out[:,-1,:].squeeze(1)
        # out = self.fc(out)
        # out = torch.sigmoid(out)
        # new version
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out

class FocalLoss(torch.nn.Module):
  """
  二分类的Focalloss alpha 固定
  """
  def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
    super().__init__()
    self.gamma = gamma
    self.alpha = alpha
    self.reduction = reduction
  
  def forward(self, pred, target):
    alpha = self.alpha
    loss = - alpha * (1 - pred) ** self.gamma * target * torch.log(pred) - \
        (1 - alpha) * pred ** self.gamma * (1 - target) * torch.log(1 - pred)
    if self.reduction == 'elementwise_mean':
      loss = torch.mean(loss)
    elif self.reduction == 'sum':
      loss = torch.sum(loss)
    return loss