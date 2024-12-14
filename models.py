import torch
from sparsemax import Sparsemax
import torch.nn as nn
import torch.nn.functional as F

#build a model with 2 inputs, 5 hidden neurons, and 2 outputs
class Net(torch.nn.Module):
    def __init__(self, dimin=2, numneuro=5, dimout=2, nonlinearity='relu', topk=None):
        super(Net, self).__init__()
        self.nonlinearity = nonlinearity
        self.numneuro = numneuro
        self.dimin = dimin
        if nonlinearity=='relu' or nonlinearity=='topk':
            lambda_pre = softplus_inverse(1/(numneuro*dimin))
        else:
            lambda_pre = softplus_inverse(1/(4*dimin))
        self.lambda_pre = nn.Parameter(lambda_pre) #trainable parameter
        if topk is not None:
            self.topk = topk
        self.fc1 = torch.nn.Linear(dimin, numneuro)
        self.fc2 = torch.nn.Linear(numneuro, dimout)

    @property
    def lambda_val(self): #lambda_val is lambda, forced to be positive here
        return F.softplus(self.lambda_pre)

    def forward(self, x, return_hidden=False):
        # lam = 1/(4*self.dimin)
        lam = self.lambda_val
        if self.nonlinearity=='relu':
            fact = nn.ReLU()
            xint = fact(lam*self.fc1(x))
        elif self.nonlinearity=='topk':
            xint = self.fc1(x)
            _, topk_indices = torch.topk(xint, self.topk, dim=-1)
            mask = torch.zeros_like(xint)
            mask.scatter_(-1, topk_indices, 1)
            xint = xint * mask* lam
        elif self.nonlinearity=='sparsemax_lintx':
            xint = self.fc1(x)
            sm = Sparsemax(dim=-1)
            xint = sm(lam*xint)
        elif self.nonlinearity=='sparsemax_dist':
            A = self.fc1.weight
            b = self.fc1.bias
            xint = -lam*torch.square(torch.norm(x.unsqueeze(1)-A.unsqueeze(0), dim=-1))
            sm = Sparsemax(dim=-1)
            xint = sm(xint)
        x = self.fc2(xint)
        if not return_hidden:
            return x
        else:
            return x, xint

def softplus_inverse(input, beta=1.0, threshold=20.0):
        """"
        inverse of the softplus function in torch
        """
        if isinstance(input, float):
                input = torch.tensor([input])
        if input*beta<threshold:
                return (1/beta)*torch.log(torch.exp(beta*input)-1.0)
        else:
              return input[0]