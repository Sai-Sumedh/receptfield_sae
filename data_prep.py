from torch.utils.data import Dataset
import numpy as np
import random
import torch
from torch.utils.data import DataLoader

class AnyDataset(Dataset):
    """
    Dataset wrapper
    """

    def __init__(self, root, dim=2, train=True, dataname=None):
        self.root = root
        self.train = train #train data or test
        self.dim = dim
        
        if train:
            filename = f"traindata.pt"
        else:
            filename = f"testdata.pt"
        datapath = root+f'/{dataname}features/'+filename

        file = torch.load(datapath)

        self.data = file['data']
        self.labels = file['labels']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx, :], self.labels[idx]


#load data


def seed_worker(worker_id):
        """
        Utility function for reproducibility in Dataloader behavior
        """
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

def loaddata(DATASET, root="./data", batch_size=64, normalize=True, return_data=False):
    traindata = AnyDataset(root, train=True, dataname=DATASET)
    testdata = AnyDataset(root, train=False, dataname=DATASET)
    if normalize:
        mean = torch.mean(traindata.data, dim=0, keepdims=True)
        traindata.data = traindata.data - mean
        testdata.data = testdata.data-mean
        datadim = traindata.data.shape[-1]
        meansqnorm = torch.mean(torch.sum(traindata.data**2, dim=-1), dim=0)
        scaling = torch.sqrt(datadim/meansqnorm)
        traindata.data = traindata.data*scaling
        testdata.data = testdata.data*scaling
    
    g_tr = torch.Generator()
    g_tr.manual_seed(0)

    g_te = torch.Generator()
    g_te.manual_seed(0)
    #create mini-batches
    train_dataloader = DataLoader(traindata, batch_size=batch_size, \
                                  worker_init_fn=seed_worker, generator=g_tr)
    test_dataloader = DataLoader(testdata, batch_size=batch_size, \
                                 worker_init_fn=seed_worker, generator=g_te)
    if not return_data:
        return train_dataloader, test_dataloader
    else:
        return train_dataloader, test_dataloader, traindata, testdata