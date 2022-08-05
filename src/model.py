import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self,input_dim,hid_dim,output_dim,num_layers,device):
        super(Model,self).__init__()
        self.device = device
        self.rnn = nn.LSTM(input_size=input_dim,hidden_size=hid_dim,num_layers=num_layers,batch_first=True)
        self.fc = nn.Linear(hid_dim,output_dim)

    def forward(self,x):
        # x = [b,seq,f],
        assert torch.isnan(x).any() == False
        out,_ = self.rnn(x)
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

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, device, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        assert torch.isnan(x).any() == False
        out, h = self.gru(x)
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out
    
    # def init_hidden(self, batch_size):
    #     weight = next(self.parameters()).data
    #     hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
    #     return hidden

class RNNNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, device, drop_prob=0.2):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        
        self.rnn = nn.RNN(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, h = self.rnn(x)
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out
    
    # def init_hidden(self, batch_size):
    #     weight = next(self.parameters()).data
    #     hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
    #               weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
    #     return hidden