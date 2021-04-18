import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NN(nn.Module):
    def __init__(self,NFILT=32,NOUT=4):
        super(NN,self).__init__()
        self.h1 = nn.Linear(9546, NFILT)
        # self.h2 = nn.Linear(NFILT, 10)
        self.out = nn.Linear(NFILT,NOUT)

    def forward(self, x):
        
        x = F.relu(self.h1(x))
        # x = F.relu(self.h2(x))
        # x = nn.Linear(32, )
        x = self.out(x)
        return F.log_softmax(x, dim=1)
