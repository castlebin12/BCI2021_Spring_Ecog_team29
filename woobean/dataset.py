import copy
import scipy.signal as signal
import scipy.stats as stats
import scipy.io as sio
import tqdm
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, path):
        self.path = path
        if self.path[-1] != '/':
            self.path += '/'        
        self.NFFF = 240

    def __len__(self):
        return 179

    def __getitem__(self, idx):
        data_set = np.load(f'{self.path}{idx}.0.npy')
        _,_, spec_data1 = signal.spectrogram(data_set[0,:],fs=1200,nperseg=64,noverlap=32,nfft=256)

        spec_data1 = spec_data1[:self.NFFF,:]
        data = spec_data1
        data = data.flatten()
        target = data_set[1][0]

        return data,target