
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

epochs = 5
lr = 0.01
momentum = 0.5
log_interval = 200
seed = 1
torch.manual_seed(seed)

# create training and validation dataset
dataset_train, dataset_valid = random_split(Dataset('./pca1_data/'), [149,30], generator=torch.Generator().manual_seed(42))
dataset_train, dataset_test = random_split(dataset_train, [121,28], generator=torch.Generator().manual_seed(42))
NWORKERS = 24
device = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN = DataLoader(dataset=dataset_train,
                   batch_size=3,
                   shuffle=True,
                   drop_last=False,
                   num_workers=NWORKERS)

TEST = DataLoader(dataset=dataset_test,
                   batch_size=3,
                   shuffle=True,
                   drop_last=False,
                   num_workers=NWORKERS)

VALID = DataLoader(dataset=dataset_valid,
                   batch_size=3,
                   shuffle=True,
                   drop_last=False,
                   num_workers=NWORKERS)


# if __name__ == "__main__":
model = NN().to(device)

optimizer = optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
loss = nn.CrossEntropyLoss()

W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
for epoch in range(2):
    model.train()
    for i,(x,t) in enumerate(TRAIN):
        optimizer.zero_grad()
        x = x.to(device).float()
        t = t.to(device).long()
        
        y = model(x)
        # y = x * W + b

        print(y)
        # J = loss(input=y,target=t)
        # cost = torch.mean((y - t) **2)
        cost = F.mse_loss(y[:,-1:,:], t)
        cost.backward()
        optimizer.step()

        if i%50==0:
            print('EPOCH:{}\tITER:{}\tLOSS:{}'.format(str(epoch).zfill(2),
                                                        str(i).zfill(5),
                                                        cost.data.cpu().numpy()))

    # evaluate results for validation test
    model.eval()
    for i,(x,t) in enumerate(TEST):
        x = x.to(device).float()
        t = t.to(device).long()
        y = model(x)
        


