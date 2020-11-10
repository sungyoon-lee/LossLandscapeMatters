import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.nn.functional as F
from torch import autograd

from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR, MultiStepLR

def pgd_l2(model_eval, X, y, epsilon=36/255, niters=100, alpha=9/255):   
    EPS = 1e-24
    X_pgd = Variable(X.data, requires_grad=True)
    
    for i in range(niters): 
        opt = optim.Adam([X_pgd], lr=1.)
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model_eval(X_pgd,method_opt="forward"), y)
        loss.backward()
        grad = 1e10*X_pgd.grad.data
                
        grad_norm = grad.view(grad.shape[0],-1).norm(2, dim=-1, keepdim=True)
        grad_norm = grad_norm.view(grad_norm.shape[0],grad_norm.shape[1],1,1)
                    
        eta = alpha*grad/(grad_norm+EPS)
        eta_norm = eta.view(eta.shape[0],-1).norm(2,dim=-1)
         
        
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = X_pgd.data-X.data                           
        mask = eta.view(eta.shape[0], -1).norm(2, dim=1) <= epsilon
        
        scaling_factor = eta.view(eta.shape[0],-1).norm(2,dim=-1)+EPS
        scaling_factor[mask] = epsilon
        
        eta *= epsilon / (scaling_factor.view(-1, 1, 1, 1)) 

        X_pgd = torch.clamp(X.data + eta, 0, 1)
        X_pgd = Variable(X_pgd.data, requires_grad=True)          
   
    return X_pgd.data
              

def pgd(model_eval, X, y, epsilon=8/255, niters=100, alpha=2/255): 
    X_pgd = Variable(X.data, requires_grad=True)
    for i in range(niters): 
        opt = optim.Adam([X_pgd], lr=1.)
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model_eval(X_pgd,method_opt="forward"), y)
        loss.backward()
        eta = alpha*X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        
        X_pgd = torch.clamp(X.data + eta, 0, 1)
        X_pgd = Variable(X_pgd, requires_grad=True)          
        
    return X_pgd.data


def pgd_(model_eval, X, y, epsilon=8/255, niters=20, alpha=2/255): #### CIFAR normalized version    
#     mean = torch.tensor([0.4914, 0.4822, .4465], dtype=torch.float32).cuda()
#     std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32).cuda()    
    mean = torch.tensor([0.43768206, 0.44376972, 0.47280434] , dtype=torch.float32).cuda()
    std = torch.tensor([0.19803014, 0.20101564, 0.19703615], dtype=torch.float32).cuda()
    
    dat = std.view(1,-1,1,1)*X.clone()+mean.view(1,-1,1,1) ## 01 space
    X_pgd_01 = dat.clone()
        
    X_pgd = Variable(X.data, requires_grad=True) ## norm'd space
    
    for i in range(niters): 
        opt = optim.Adam([X_pgd], lr=1.) ## norm'd space
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model_eval(X_pgd,method_opt="forward"), y) ## norm'd space
        loss.backward()
        eta = alpha*X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd_01.data + eta, requires_grad=True) ## 01 space

        eta = torch.clamp(X_pgd.data - dat.data, -epsilon, epsilon) ## 01 space
        
        X_pgd = torch.clamp(dat.data + eta, 0, 1) ## 01 space
        X_pgd_01 = X_pgd.clone()
        X_pgd = (X_pgd-mean.view(1,-1,1,1))/std.view(1,-1,1,1) ## norm'd space
        
        X_pgd = Variable(X_pgd.data, requires_grad=True)          
 
    return X_pgd.data




def evaluate_pgd(loader, model,norm, epsilon=2/255, alpha=0.5/255, niters=100):
    losses = AverageMeter()
    errors = AverageMeter()

    model.eval()

    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda()
        if norm== np.inf:
            X_pgd = pgd(model, X, y, epsilon, niters, alpha)
            #X_pgd = pgd_(model, X, y, epsilon, niters, alpha) ##unnormalize
        #    print("YYYYYYYYYYYYYYYY",y)
        else:
            X_pgd = pgd_l2(model, X, y, epsilon, niters, alpha)
            
        out = model(Variable(X_pgd),method_opt="forward")
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)
        #print("ERRRRRRRRRRRRRRRR",err)
        losses.update(ce.data, X.size(0))
        errors.update(err, X.size(0))
    print(' * Error {error.avg:.3f}'
          .format(error=errors))
    return errors.avg

def evaluate_pgd_n(loader, model,norm, epsilon=2/255, alpha=0.5/255, niters=100):
    losses = AverageMeter()
    errors = AverageMeter()

    model.eval()

    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda()
        if norm== np.inf:
            #X_pgd = pgd(model, X, y, epsilon, niters, alpha)
            X_pgd = pgd_(model, X, y, epsilon, niters, alpha) ##unnormalize
        #    print("YYYYYYYYYYYYYYYY",y)
        else:
            X_pgd = pgd_l2(model, X, y, epsilon, niters, alpha)
            
        out = model(Variable(X_pgd),method_opt="forward")
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.data.max(1)[1] != y).float().sum()  / X.size(0)
        #print("ERRRRRRRRRRRRRRRR",err)
        losses.update(ce.data, X.size(0))
        errors.update(err, X.size(0))
    print(' * Error {error.avg:.3f}'
          .format(error=errors))
    return errors.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
