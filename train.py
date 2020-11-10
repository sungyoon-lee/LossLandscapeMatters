
import sys
import copy
import torch
from torch.nn import Sequential, Linear, ReLU, CrossEntropyLoss
import numpy as np
from datasets import loaders
from bound_layers import BoundSequential, BoundLinear, BoundConv2d, BoundDataParallel
import torch.optim as optim
# from gpu_profile import gpu_profile
import time
from datetime import datetime
# from convex_adversarial import DualNetwork
from eps_scheduler import EpsilonScheduler
from config import load_config, get_path, config_modelloader, config_dataloader, update_dict
from argparser import argparser
from pgd_eval import evaluate_pgd, evaluate_pgd_n
# sys.settrace(gpu_profile)

#DEBUGG = False True
DEBUGG = False
BREAK = False


#from multi_eval import Multi_eval 

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

class Logger(object):
    def __init__(self, log_file = None, log_file_loss = None, log_file_grad=None, log_file_grad_norm=None, log_file_a_sign=None , log_file_cosine=None, log_file_loss_max=None):
        self.log_file = log_file
        self.log_file_loss = log_file_loss
        self.log_file_grad = log_file_grad
        self.log_file_grad_norm = log_file_grad_norm
        self.log_file_a_sign = log_file_a_sign
        self.log_file_cosine = log_file_cosine
        self.log_file_loss_max = log_file_loss_max

    def log(self, *args, **kwargs):
        print(*args, **kwargs)
        if self.log_file:
            print(*args, **kwargs, file = self.log_file)
            self.log_file.flush()
    def loss(self, *args, **kwargs):
        if self.log_file_loss:
            print(*args, **kwargs, file = self.log_file_loss)
            self.log_file_loss.flush()
    def loss_max(self, *args, **kwargs):
        if self.log_file_loss_max:
            print(*args, **kwargs, file = self.log_file_loss_max)
            self.log_file_loss_max.flush()
    def grad(self, *args, **kwargs):
        if self.log_file_grad:
            print(*args, **kwargs, file = self.log_file_grad)
            self.log_file_grad.flush()
    def grad_norm(self, *args, **kwargs):
        if self.log_file_grad_norm:
            print(*args, **kwargs, file = self.log_file_grad_norm)
            self.log_file_grad_norm.flush()
    def a_sign(self, *args, **kwargs):
        if self.log_file_a_sign:
            print(*args, **kwargs, file = self.log_file_a_sign)
            self.log_file_a_sign.flush()
    def cosine(self, *args, **kwargs):
        if self.log_file_cosine:
            print(*args, **kwargs, file = self.log_file_cosine)
            self.log_file_cosine.flush()

            

    
    
    
    
def Train_calloss(model, t, i, data, labels, loader, eps,end_eps, max_eps, norm, logger, verbose, train, method, cal_grad = False, lower_d_list2=[],beta=1.0, kappa=0.0,cal_lb=False, frozen =5, **kwargs):      

    # if train=True, use training mode
    # if train=False, use test mode, no back prop
    
    num_class = 10
    batch_multiplier = kwargs.get("batch_multiplier", 1)  
    if cal_grad:
        model.train() 
    else:
        model.eval()
    # pregenerate the array for specifications, will be used for scatter
    sa = np.zeros((num_class, num_class - 1), dtype = np.int32)
    for ii in range(sa.shape[0]):
        for j in range(sa.shape[1]):
            if j < ii:
                sa[ii][j] = j
            else:
                sa[ii][j] = j + 1
    sa = torch.LongTensor(sa) 
    batch_size = loader.batch_size * batch_multiplier
    if batch_multiplier > 1 and train:
        logger.log('Warning: Large batch training. The equivalent batch size is {} * {} = {}.'.format(batch_multiplier, loader.batch_size, batch_size))
    # per-channel std and mean
    std = torch.tensor(loader.std).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    mean = torch.tensor(loader.mean).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    model_range = 0.0
    #end_eps = eps_scheduler.get_eps(t+1, 0)
    if end_eps < np.finfo(np.float32).tiny:
        method = "natural"
    start = time.time()
    # generate specifications
    c = torch.eye(num_class).type_as(data)[labels].unsqueeze(1) - torch.eye(num_class).type_as(data).unsqueeze(0) 
    # remove specifications to self
    I = (~(labels.data.unsqueeze(1) == torch.arange(num_class).type_as(labels.data).unsqueeze(0)))
    c = (c[I].view(data.size(0),num_class-1,num_class))
    # scatter matrix to avoid compute margin to self
    sa_labels = sa[labels]
    # storing computed lower bounds after scatter
    lb_s = torch.zeros(data.size(0), num_class)
    ub_s = torch.zeros(data.size(0), num_class)

    # FIXME: Assume unnormalized data is from range 0 - 1
    if kwargs["bounded_input"]:
        if norm != np.inf:
            raise ValueError("bounded input only makes sense for Linf perturbation. "
                             "Please set the bounded_input option to false.")
        data_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
        data_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))
        data_ub = torch.min(data + (eps / std), data_max)
        data_lb = torch.max(data - (eps / std), data_min)
    else:
        if norm == np.inf:
            data_ub = data.cpu() + (eps / std)
            data_lb = data.cpu() - (eps / std)
        else:
            # For other norms, eps will be used instead.
            data_ub = data_lb = data

    if list(model.parameters())[0].is_cuda:
        data = data.cuda()
        data_ub = data_ub.cuda()
        data_lb = data_lb.cuda()
        labels = labels.cuda()
        c = c.cuda()
        sa_labels = sa_labels.cuda()
        lb_s = lb_s.cuda()
        ub_s = ub_s.cuda()
    # convert epsilon to a tensor
    eps_tensor = data.new(1)
    eps_tensor[0] = eps

    # omit the regular cross entropy, since we use robust error
    output = model(data, method_opt="forward", disable_multi_gpu = (method == "natural"))
    regular_ce = CrossEntropyLoss()(output, labels)
    # get range statistic
    model_range = output.max().detach().cpu().item() - output.min().detach().cpu().item()

    if verbose or method != "natural":
        if kwargs["bound_type"] == "convex-adv":
            # Wong and Kolter's bound, or equivalently Fast-Lin
            if kwargs["convex-proj"] is not None:
                proj = kwargs["convex-proj"]
                if norm == np.inf:
                    norm_type = "l1_median"
                    norm_type = "l2_normal"
                else:
                    raise(ValueError("Unsupported norm {} for convex-adv".format(norm)))
            else:
                proj = None
                if norm == np.inf:
                    norm_type = "l1"
                elif norm == 2:
                    norm_type = "l2"
                else:
                    raise(ValueError("Unsupported norm {} for convex-adv".format(norm)))
            if loader.std == [1] or loader.std == [1, 1, 1]:
                convex_eps = eps
            else:
                convex_eps = eps / np.mean(loader.std)
                # for CIFAR we are roughly / 0.2
                # FIXME this is due to a bug in convex_adversarial, we cannot use per-channel eps
            if norm == np.inf:
                # bounded input is only for Linf
                if kwargs["bounded_input"]:
                    # FIXME the bounded projection in convex_adversarial has a bug, data range must be positive
                    assert loader.std == [1,1,1] or loader.std == [1]
                    data_l = 0.0
                    data_u = 1.0
                else:
                    data_l = -np.inf
                    data_u = np.inf
            else:
                data_l = data_u = None
            f = DualNetwork(model, data, convex_eps, proj = proj, norm_type = norm_type, bounded_input = kwargs["bounded_input"], data_l = data_l, data_u = data_u)
            lb = f(c)
        elif kwargs["bound_type"] == "interval":
            ub, lb, relu_activity, unstable, dead, alive = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, method_opt="interval_range")
        elif kwargs["bound_type"] == "crown-full":
            _, _, lb, _ = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, upper=False, lower=True, method_opt="full_backward_range")
            unstable = dead = alive = relu_activity = torch.tensor([0])
        elif kwargs["bound_type"] == "crown-interval":
            # Enable multi-GPU only for the computationally expensive CROWN-IBP bounds, 
            # not for regular forward propagation and IBP because the communication overhead can outweigh benefits, giving little speedup. 
            
            ub, ilb, relu_activity, unstable, dead, alive = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, method_opt="interval_range")
            if beta < 1e-5:
                lb = ilb
            else:
                if kwargs["runnerup_only"]:
                    # regenerate a smaller c, with just the runner-up prediction
                    # mask ground truthlabel output, select the second largest class
                    masked_output = output.detach().scatter(1, labels.unsqueeze(-1), -100)
                    # location of the runner up prediction
                    runner_up = masked_output.max(1)[1]
                    # get margin from the groud-truth to runner-up only
                    runnerup_c = torch.eye(num_class).type_as(data)[labels]
                    runnerup_c.scatter_(1, runner_up.unsqueeze(-1), -1)
                    runnerup_c = runnerup_c.unsqueeze(1).detach()
                    # get the bound for runnerup_c
                    _, _, clb, bias = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, method_opt="backward_range")
                    clb = clb.expand(clb.size(0), num_class - 1)
                else:
                    # get the CROWN bound using interval bounds 
                    _, _, clb, bias = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, method_opt="backward_range",lower_d_list2=lower_d_list2)
                # how much better is crown-ibp better than ibp?
                diff = (clb - ilb).sum().item()
                lb = clb * beta + ilb * (1 - beta)
        elif kwargs["bound_type"] == "crown-interval-frozen": 
            # Enable multi-GPU only for the computationally expensive CROWN-IBP bounds, 
            # not for regular forward propagation and IBP because the communication overhead can outweigh benefits, giving little speedup. 

            ub, ilb, relu_activity, unstable, dead, alive = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, method_opt="interval_range")

            if beta < 1e-5:
                lb = ilb
            else:
                # get the CROWN bound using interval bounds 
                _, _, clb, bias = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, method_opt="backward_range_frozen",lower_d_list2=lower_d_list2, frozen=frozen)

                diff = (clb - ilb).sum().item()
                lb = clb * beta + ilb * (1 - beta)
        else:
            raise RuntimeError("Unknown bound_type " + kwargs["bound_type"]) 
        lb = lb_s.scatter(1, sa_labels, lb)
        robust_ce = CrossEntropyLoss()(-lb, labels)
    if method == "robust":
        loss = robust_ce
    elif method == "robust_activity":
        loss = robust_ce + kwargs["activity_reg"] * relu_activity.sum()
    elif method == "natural":
        loss = regular_ce
        robust_ce = loss
    elif method == "robust_natural":
        loss = (1-kappa) * robust_ce + kappa * regular_ce
    else:
        raise ValueError("Unknown method " + method)

    if train and kwargs["l1_reg"] > np.finfo(np.float32).tiny:
        reg = kwargs["l1_reg"]
        l1_loss = 0.0
        for name, param in model.named_parameters():
            if 'bias' not in name:
                l1_loss = l1_loss + torch.sum(torch.abs(param))
        l1_loss = reg * l1_loss
        loss = loss + l1_loss
        
    if cal_lb:
        return loss, robust_ce, clb, bias
    else:
        return loss, robust_ce
            
            
def Train(model, t, loader, eps_scheduler, max_eps, norm, logger, verbose, train, opt, method, cal_loss=False, n_loss=2,max_loss=4, min_loss=0.5, cal_grad = False, config = None, cal_grad_norm=False,start_beta = 1.0,start_kappa=1.0, lr_consist=False,bound_opt_eval=None, frozen=13, **kwargs):
    # if train=True, use training mode
    # if train=False, use test mode, no back prop
    if model.bound_opts['ours'] and train is False :
        torch.set_grad_enabled(True)
    
    
    num_class = 10
    losses = AverageMeter()
    l1_losses = AverageMeter()
    errors = AverageMeter()
    robust_errors = AverageMeter()
    regular_ce_losses = AverageMeter()
    robust_ce_losses = AverageMeter()
    relu_activities = AverageMeter()
    bound_bias = AverageMeter()
    bound_diff = AverageMeter()
    unstable_neurons = AverageMeter()
    dead_neurons = AverageMeter()
    alive_neurons = AverageMeter()
    batch_time = AverageMeter()
    batch_multiplier = kwargs.get("batch_multiplier", 1)  
    
    

    loss_points = np.logspace(np.log10(min_loss), np.log10(max_loss), n_loss, endpoint=True)
    loss_points = np.concatenate((np.zeros(1),loss_points,np.ones(1)), axis=0)

    
    ReLU_lower_bs = AverageMeter()
    not_ReLU_lower_bs = AverageMeter()
    A_xs = AverageMeter()
    A_norms = AverageMeter()

    if train:
        model.train() 
    else:
        model.eval()
    # pregenerate the array for specifications, will be used for scatter
    sa = np.zeros((num_class, num_class - 1), dtype = np.int32)
    for i in range(sa.shape[0]):
        for j in range(sa.shape[1]):
            if j < i:
                sa[i][j] = j
            else:
                sa[i][j] = j + 1
    sa = torch.LongTensor(sa) 
    batch_size = loader.batch_size * batch_multiplier
    if batch_multiplier > 1 and train:
        logger.log('Warning: Large batch training. The equivalent batch size is {} * {} = {}.'.format(batch_multiplier, loader.batch_size, batch_size))
    # per-channel std and mean
    std = torch.tensor(loader.std).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    
    mean = torch.tensor(loader.mean).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    model_range = 0.0
    end_eps = eps_scheduler.get_eps(t+1, 0)
    if end_eps < np.finfo(np.float32).tiny:
        logger.log('eps {} close to 0, using natural training'.format(end_eps))
        method = "natural"
    for i, (data, labels) in enumerate(loader): 
        if i>0 and BREAK:
            break
#        original_A_sign = torch.zeros_like(data)
        start = time.time()
        eps = eps_scheduler.get_eps(t, int(i//batch_multiplier)) 
        crown_final_beta = kwargs['final-beta'] 
        natural_final_factor = kwargs["final-kappa"] 
        beta = start_beta - (1.0-(max_eps-eps)/max_eps)*(start_beta-crown_final_beta) 
        kappa = start_kappa - (1.0-(max_eps-eps)/max_eps)*(start_kappa-natural_final_factor) 
        
        if train and i % batch_multiplier == 0:
            opt.zero_grad()
        # generate specifications
        c = torch.eye(num_class).type_as(data)[labels].unsqueeze(1) - torch.eye(num_class).type_as(data).unsqueeze(0) 
        # remove specifications to self
        I = (~(labels.data.unsqueeze(1) == torch.arange(num_class).type_as(labels.data).unsqueeze(0)))
        c = (c[I].view(data.size(0),num_class-1,num_class))
        # scatter matrix to avoid compute margin to self
        sa_labels = sa[labels]
        # storing computed lower bounds after scatter
        lb_s = torch.zeros(data.size(0), num_class)
        ub_s = torch.zeros(data.size(0), num_class)

        # FIXME: Assume unnormalized data is from range 0 - 1
        if kwargs["bounded_input"]:
            if norm != np.inf:
                raise ValueError("bounded input only makes sense for Linf perturbation. "
                                 "Please set the bounded_input option to false.")
            data_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
            data_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))
            data_ub = torch.min(data + (eps / std), data_max)
            data_lb = torch.max(data - (eps / std), data_min)
        else:
            if norm == np.inf:
                data_ub = data + (eps / std)
                data_lb = data - (eps / std)
            else:
                # For other norms, eps will be used instead.
                data_ub = data_lb = data

        if list(model.parameters())[0].is_cuda:
            data = data.cuda()
            data_ub = data_ub.cuda()
            data_lb = data_lb.cuda()
            labels = labels.cuda()
            c = c.cuda()
            sa_labels = sa_labels.cuda()
            lb_s = lb_s.cuda()
            ub_s = ub_s.cuda()
        # convert epsilon to a tensor
        eps_tensor = data.new(1)
        eps_tensor[0] = eps
        

        # omit the regular cross entropy, since we use robust error
        output = model(data, method_opt="forward", disable_multi_gpu = (method == "natural"))
        regular_ce = CrossEntropyLoss()(output, labels)
        regular_ce_losses.update(regular_ce.cpu().detach().numpy(), data.size(0))
        errors.update(torch.sum(torch.argmax(output, dim=1)!=labels).cpu().detach().numpy()/data.size(0), data.size(0))
        # get range statistic
        model_range = output.max().detach().cpu().item() - output.min().detach().cpu().item()

        if verbose or method != "natural":
            if kwargs["bound_type"] == "convex-adv":
                # Wong and Kolter's bound, or equivalently Fast-Lin
                if kwargs["convex-proj"] is not None:
                    proj = kwargs["convex-proj"]
                    if norm == np.inf:
                        norm_type = "l1_median"
                    elif norm == 2:
                        norm_type = "l2_normal"
                    else:
                        raise(ValueError("Unsupported norm {} for convex-adv".format(norm)))
                else:
                    proj = None
                    if norm == np.inf:
                        norm_type = "l1"
                    elif norm == 2:
                        norm_type = "l2"
                    else:
                        raise(ValueError("Unsupported norm {} for convex-adv".format(norm)))
                if loader.std == [1] or loader.std == [1, 1, 1]:
                    convex_eps = eps
                else:
                    convex_eps = eps / np.mean(loader.std)
                    # for CIFAR we are roughly / 0.2
                    # FIXME this is due to a bug in convex_adversarial, we cannot use per-channel eps
                if norm == np.inf:
                    # bounded input is only for Linf
                    if kwargs["bounded_input"]:
                        # FIXME the bounded projection in convex_adversarial has a bug, data range must be positive
                        assert loader.std == [1,1,1] or loader.std == [1]
                        data_l = 0.0
                        data_u = 1.0
                    else:
                        data_l = -np.inf
                        data_u = np.inf
                else:
                    data_l = data_u = None
                f = DualNetwork(model, data, convex_eps, proj = proj, norm_type = norm_type, bounded_input = kwargs["bounded_input"], data_l = data_l, data_u = data_u)
                lb = f(c)
            elif kwargs["bound_type"] == "interval":
                ub, lb, relu_activity, unstable, dead, alive = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, method_opt="interval_range")
            elif kwargs["bound_type"] == "crown-full":
                _, _, lb, _ = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, upper=False, lower=True, method_opt="full_backward_range")
                unstable = dead = alive = relu_activity = torch.tensor([0])
            elif kwargs["bound_type"] == "crown-interval":
                # Enable multi-GPU only for the computationally expensive CROWN-IBP bounds, 
                # not for regular forward propagation and IBP because the communication overhead can outweigh benefits, giving little speedup. 
                ub, ilb, relu_activity, unstable, dead, alive = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, method_opt="interval_range")

                if beta < 1e-5:
                    lb = ilb
                else:
                    if kwargs["runnerup_only"]:
                        # regenerate a smaller c, with just the runner-up prediction
                        # mask ground truthlabel output, select the second largest class
                        masked_output = output.detach().scatter(1, labels.unsqueeze(-1), -100)
                        # location of the runner up prediction
                        runner_up = masked_output.max(1)[1]
                        # get margin from the groud-truth to runner-up only
                        runnerup_c = torch.eye(num_class).type_as(data)[labels]
                        # set the runner up location to -
                        runnerup_c.scatter_(1, runner_up.unsqueeze(-1), -1)
                        runnerup_c = runnerup_c.unsqueeze(1).detach()
                        # get the bound for runnerup_c
                        _, _, clb, bias = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, method_opt="backward_range")
                        clb = clb.expand(clb.size(0), num_class - 1)
                    else:
                        # get the CROWN bound using interval bounds 
                        _, _, clb, bias = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, method_opt="backward_range")
 #                       bound_bias.update(bias.sum() / data.size(0))
                    lb = clb * beta + ilb * (1 - beta)
 #                  
            elif kwargs["bound_type"] == "crown-interval-frozen":
                # Enable multi-GPU only for the computationally expensive CROWN-IBP bounds, 
                # not for regular forward propagation and IBP because the communication overhead can outweigh benefits, giving little speedup. 

                ub, ilb, relu_activity, unstable, dead, alive = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, method_opt="interval_range")
                
                if beta < 1e-5:
                    lb = ilb
                else:
                    _, _, clb, bias = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, method_opt="backward_range_frozen", frozen=frozen)
                    diff = (clb - ilb).sum().item()
                    lb = clb * beta + ilb * (1 - beta)
            else:
                raise RuntimeError("Unknown bound_type " + kwargs["bound_type"]) 
            lb = lb_s.scatter(1, sa_labels, lb)
            robust_ce = CrossEntropyLoss()(-lb, labels)
            if kwargs["bound_type"] != "convex-adv":
                
                relu_activities.update(relu_activity.sum().detach().cpu().item() / data.size(0), data.size(0))
                unstable_neurons.update(unstable.sum().detach().cpu().item() / data.size(0), data.size(0))
                dead_neurons.update(dead.sum().detach().cpu().item() / data.size(0), data.size(0))
                alive_neurons.update(alive.sum().detach().cpu().item() / data.size(0), data.size(0))


            
        if method == "robust":
            loss = robust_ce
        elif method == "robust_activity":
            loss = robust_ce + kwargs["activity_reg"] * relu_activity.sum()
        elif method == "natural":
            loss = regular_ce
            robust_ce = loss
        elif method == "robust_natural":
            loss = (1-kappa) * robust_ce + kappa * regular_ce
        else:
            raise ValueError("Unknown method " + method)

        if train and kwargs["l1_reg"] > np.finfo(np.float32).tiny:
            reg = kwargs["l1_reg"]
            l1_loss = 0.0
            for name, param in model.named_parameters():
                if 'bias' not in name:
                    l1_loss = l1_loss + torch.sum(torch.abs(param))
            l1_loss = reg * l1_loss
            loss = loss + l1_loss
            l1_losses.update(l1_loss.cpu().detach().numpy(), data.size(0))
        
        ##update a
        if method != "natural" and model.bound_opts['ours'] and beta >= 1e-5: 
            loss, robust_ce, clb, bias=Train_calloss_ours(loss,robust_ce,model, t, i, data, labels, loader, eps,end_eps, max_eps, norm, logger, verbose, train, method, beta=beta, kappa=kappa,cal_grad=True,cal_lb=True,multistep=False,frozen=frozen, **kwargs)
                
        if (verbose or method != "natural") and kwargs["bound_type"] == "crown-interval" and beta >= 1e-5 and not kwargs["runnerup_only"]:
            bound_bias.update(bias.sum() / data.size(0))

            ReLU_lower_bs.update(model.ReLU_lower_b.sum(), data.size(0))
            if (model.ReLU_lower_b>0).sum()>0:
                print("There is positive part in the ReLU_lower_bs!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

            not_ReLU_lower_bs.update(model.not_ReLU_lower_b.sum(), data.size(0))                    
            A_norms.update(model.A_norm.sum(), data.size(0))
            A_xs.update(model.A_x.sum(), data.size(0))
            diff = (clb - ilb).sum().item()
            bound_diff.update(diff / data.size(0), data.size(0))
            lb = clb * beta + ilb * (1 - beta)
            original_A_sign = model.lower_A.sign()
            
        elif (verbose or method != "natural") and kwargs["bound_type"] == "crown-interval-frozen" and beta >= 1e-5 and not kwargs["runnerup_only"]:
            diff = (clb - ilb).sum().item()
            lb = clb * beta + ilb * (1 - beta)
    
        if train: 
            opt.zero_grad()
            loss.backward()
            
            if cal_loss and (t*len(loader)+i)%100 ==0:
                if method != "natural" and model.bound_opts['ours'] and beta >= 1e-5:
                    loss_max, Rloss_tmp= Train_calloss(model, t, i, data, labels, loader, max_eps, max_eps, max_eps, norm, logger, verbose, train, method,cal_grad=cal_grad, beta=beta,kappa=0, cal_lb=False,frozen=frozen,**kwargs)
                    loss_max, _, clb, bias=Train_calloss_ours(loss_max,Rloss_tmp,model, t, i, data, labels, loader, max_eps, max_eps, max_eps, norm, logger, verbose, train, method, beta=beta, kappa=0,cal_grad=True,cal_lb=True,multistep=False,frozen=frozen, **kwargs)
                else:
                    loss_max, _ = Train_calloss(model, t, i, data, labels, loader, max_eps, max_eps, max_eps, norm, logger, verbose, train, method,cal_grad=cal_grad, beta=beta,kappa=0, cal_lb=False,frozen=frozen,**kwargs)
                logger.loss_max(float(loss_max.data.cpu()), end='\t')

                model_list = [qq for qq in model.state_dict().keys()]
                dict_all_tmp = {}
                lr = opt.state_dict()['param_groups'][0]['lr']
                original_grad = []
                for point in loss_points:
                    dict_all_tmp[point]={}

                for v, param in enumerate(model.parameters()):
                    param_grad = param.grad.data.detach()
                    original_grad.append(param_grad.view(1,-1))
                    for point in loss_points:
                        param_tmp = param.data.detach() - point*lr*param_grad 
                        dict_all_tmp[point][model_list[v]] = param_tmp 
                        
                MSE_grad_norm = torch.zeros(1).cuda()
                for point in loss_points:
                    if config:
                        global_train_config = config["training_params"]
                        train_config = copy.deepcopy(global_train_config)
                        models, model_names = config_modelloader(config) 
                        model_loss_clean = BoundSequential.convert(models[0], train_config["method_params"]["bound_opts"])
                    model_loss = model_loss_clean.cuda()
                    model_loss.load_state_dict(dict_all_tmp[point])
                    opt_loss = optim.SGD(model_loss.parameters(), lr=0.00)
                    opt_loss.zero_grad()
                    if method != "natural" and model.bound_opts['ours'] and beta >= 1e-5:
                        
                        loss_tmp, Rloss_tmp= Train_calloss(model_loss, t, i, data, labels, loader, eps,end_eps, max_eps, norm, logger, verbose, train, method,cal_grad=cal_grad, beta=beta,kappa=kappa, cal_lb=False,frozen=frozen,**kwargs)
                        loss_tmp, Rloss_tmp, clb, bias=Train_calloss_ours(loss_tmp,Rloss_tmp,model_loss, t, i, data, labels, loader, eps,end_eps, max_eps, norm, logger, verbose, train, method, beta=beta, kappa=kappa,cal_grad=True,cal_lb=True,multistep=False,frozen=frozen, **kwargs)
                        
                        
                    else:
                        loss_tmp, Rloss_tmp= Train_calloss(model_loss, t, i, data, labels, loader, eps,end_eps, max_eps, norm, logger, verbose, train, method,cal_grad=cal_grad, beta=beta,kappa=kappa, cal_lb=False,frozen=frozen,**kwargs)
                    loss_tmp.backward()
                    logger.loss(float(loss_tmp.data.cpu()), end='\t')
                    model_loss_grad = []
                    if cal_grad:
                        for v, param_loss in enumerate(model_loss.parameters()):
                            model_loss_grad.append(param_loss.grad.data.view(1,-1)) 
                    MSE_grad_tmp = torch.nn.MSELoss(reduction='mean')(torch.cat(original_grad,dim=1),torch.cat(model_loss_grad,dim=1)).sqrt()
                    cosine_grad_tmp = torch.nn.CosineSimilarity(dim=1)(torch.cat(original_grad,dim=1),torch.cat(model_loss_grad,dim=1))                    
                    if method != "natural" and kwargs["bound_type"] == "crown-interval" and beta >= 1e-5:
                        A_sign_tmp = torch.nn.L1Loss()( original_A_sign , model_loss.lower_A.sign())
                        logger.a_sign(float(A_sign_tmp.data.cpu()), end='\t')
                    if cal_grad_norm:
                        MSE_grad_norm = torch.max(MSE_grad_tmp/(original_grad[0].norm()*lr*point),MSE_grad_norm)
                    
                    logger.grad(float(MSE_grad_tmp.data.cpu()), end='\t')
                    logger.cosine(float(cosine_grad_tmp.data.cpu()), end='\t')        

            
                del model_loss
                del MSE_grad_tmp, loss_tmp, Rloss_tmp
                del model_loss_clean
                if cal_grad_norm:
                    logger.grad_norm(float(MSE_grad_norm.data.cpu()), end='\t')
                logger.loss('\n', end='')
                logger.grad('\n', end='')
                logger.cosine('\n', end='')
                logger.grad_norm('\n', end='') 
                if method != "natural" and kwargs["bound_type"] == "crown-interval" and beta >= 1e-5:
                    logger.a_sign('\n', end='')                
                model.train()
            if i % batch_multiplier == 0 or i == len(loader) - 1:
                opt.step()     
            if cal_loss and (t*len(loader)+i)%100 ==0:   
                opt_loss.step()
                
                
        losses.update(loss.cpu().detach().numpy(), data.size(0))


        if verbose or method != "natural":
            robust_ce_losses.update(robust_ce.cpu().detach().numpy(), data.size(0))
            robust_errors.update(torch.sum((lb<0).any(dim=1)).cpu().detach().numpy() / data.size(0), data.size(0))

        batch_time.update(time.time() - start)
        all_val = ReLU_lower_bs.val+A_xs.val+not_ReLU_lower_bs.val+A_norms.val
        all_avg = ReLU_lower_bs.avg+A_xs.avg+not_ReLU_lower_bs.avg+A_norms.avg        
        if i % 50 == 0 and train:
            logger.log(  '[{:2d}:{:4d}]: eps {:6f}  '
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Total Loss {loss.val:.4f} ({loss.avg:.4f})  '
                    'L1 Loss {l1_loss.val:.4f} ({l1_loss.avg:.4f})  '
                    'CE {regular_ce_loss.val:.4f} ({regular_ce_loss.avg:.4f})  '
                    'RCE {robust_ce_loss.val:.4f} ({robust_ce_loss.avg:.4f})  '
                    'Err {errors.val:.4f} ({errors.avg:.4f})  '
                    'Rob Err {robust_errors.val:.4f} ({robust_errors.avg:.4f})  '
                    'Uns {unstable.val:.1f} ({unstable.avg:.1f})  '
                    'Dead {dead.val:.1f} ({dead.avg:.1f})  '
                    'Alive {alive.val:.1f} ({alive.avg:.1f})  '
                    'Tightness {tight.val:.5f} ({tight.avg:.5f})  '
                    'Bias {bias.val:.5f} ({bias.avg:.5f})  '
                    'Diff {diff.val:.5f} ({diff.avg:.5f})  '
                    'R {model_range:.3f}  '
                    'beta {beta:.3f} ({beta:.3f})  '
                    'kappa {kappa:.3f} ({kappa:.3f})  '
                    'A_x {Axs.val:.5f} ({Axs.avg:.5f})  ' 
                    'not_relu_b {not_relu_b.val:.5f} ({not_relu_b.avg:.5f})  '
                    'A_norm {A_norms.val:.5f} ({A_norms.avg:.5f})  '
                    'Relu_b {Relu_bs.val:.5f} ({Relu_bs.avg:.5f})  '
                    'All {all_val:.5f} ({all_avg:.5f})  '.format(
                    t, i, eps, batch_time=batch_time,
                    loss=losses, errors=errors, robust_errors = robust_errors, l1_loss = l1_losses,
                    regular_ce_loss = regular_ce_losses, robust_ce_loss = robust_ce_losses, 
                    unstable = unstable_neurons, dead = dead_neurons, alive = alive_neurons,
                    tight = relu_activities, bias = bound_bias, diff = bound_diff,
                    model_range = model_range, 
                    beta=beta, kappa = kappa,
                    Axs = A_xs, not_relu_b = not_ReLU_lower_bs, A_norms = A_norms, Relu_bs = ReLU_lower_bs , all_val=all_val, all_avg =all_avg
                    ))
    
    if(bound_opt_eval):
        if bound_opt_eval.get("ours", False):
            bound_opt_name="ours"
        elif bound_opt_eval.get("same-slope", False):
            bound_opt_name="same-slope"
        elif bound_opt_eval.get("zero-lb", False):
            bound_opt_name="zero-lb"
        elif bound_opt_eval.get("one-lb", False):
            bound_opt_name="one-lb"
        elif bound_opt_eval.get("binary", False):
            bound_opt_name="binary"
        elif bound_opt_eval.get("uniform", False):
            bound_opt_name="uniform"
        else:
            bound_opt_name="crown-ibp"
            
            
        logger.log(  '[{} epoch:{:2d} eps:{:.6f}]: '
        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
        'Total Loss {loss.val:.4f} ({loss.avg:.4f})  '
        'L1 Loss {l1_loss.val:.4f} ({l1_loss.avg:.4f})  '
        'CE {regular_ce_loss.val:.4f} ({regular_ce_loss.avg:.4f})  '
        'RCE {robust_ce_loss.val:.4f} ({robust_ce_loss.avg:.4f})  '
        'Uns {unstable.val:.3f} ({unstable.avg:.3f})  '
        'Dead {dead.val:.1f} ({dead.avg:.1f})  '
        'Alive {alive.val:.1f} ({alive.avg:.1f})  '
        'Tight {tight.val:.5f} ({tight.avg:.5f})  '
        'Bias {bias.val:.5f} ({bias.avg:.5f})  '
        'Diff {diff.val:.5f} ({diff.avg:.5f})  '
        'Err {errors.val:.4f} ({errors.avg:.4f})  '
        'Rob Err {robust_errors.val:.4f} ({robust_errors.avg:.4f})  '
        'R {model_range:.3f}  '
        'beta {beta:.3f} ({beta:.3f})  '
        'kappa {kappa:.3f} ({kappa:.3f})  '
        'A_x {Axs.val:.5f} ({Axs.avg:.5f})  '
        'not_relu_b {not_relu_b.val:.5f} ({not_relu_b.avg:.5f})  '
        'A_norm {A_norms.val:.5f} ({A_norms.avg:.5f})  '
        'Relu_b {Relu_bs.val:.5f} ({Relu_bs.avg:.5f})  '
        'All {all_val:.5f} ({all_avg:.5f}) '.format(
        bound_opt_name,t, eps, batch_time=batch_time,
        loss=losses, errors=errors, robust_errors = robust_errors, l1_loss = l1_losses,
        regular_ce_loss = regular_ce_losses, robust_ce_loss = robust_ce_losses, 
        unstable = unstable_neurons, dead = dead_neurons, alive = alive_neurons,
        tight = relu_activities, bias = bound_bias, diff = bound_diff,
        model_range = model_range, 
        kappa = kappa, beta=beta,Axs = A_xs, not_relu_b = not_ReLU_lower_bs, A_norms = A_norms, Relu_bs = ReLU_lower_bs , all_val=all_val, all_avg =all_avg
))
        
    else:
        logger.log(  '[FINAL RESULT epoch:{:2d} eps:{:.6f}]: '
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
            'Total Loss {loss.val:.4f} ({loss.avg:.4f})  '
            'L1 Loss {l1_loss.val:.4f} ({l1_loss.avg:.4f})  '
            'CE {regular_ce_loss.val:.4f} ({regular_ce_loss.avg:.4f})  '
            'RCE {robust_ce_loss.val:.4f} ({robust_ce_loss.avg:.4f})  '
            'Uns {unstable.val:.3f} ({unstable.avg:.3f})  '
            'Dead {dead.val:.1f} ({dead.avg:.1f})  '
            'Alive {alive.val:.1f} ({alive.avg:.1f})  '
            'Tight {tight.val:.5f} ({tight.avg:.5f})  '
            'Bias {bias.val:.5f} ({bias.avg:.5f})  '
            'Diff {diff.val:.5f} ({diff.avg:.5f})  '
            'Err {errors.val:.4f} ({errors.avg:.4f})  '
            'Rob Err {robust_errors.val:.4f} ({robust_errors.avg:.4f})  '
            'R {model_range:.3f}  '
            'beta {beta:.3f} ({beta:.3f})  '
            'kappa {kappa:.3f} ({kappa:.3f})  '
            'A_x {Axs.val:.5f} ({Axs.avg:.5f})  '
            'not_relu_b {not_relu_b.val:.5f} ({not_relu_b.avg:.5f})  '
            'A_norm {A_norms.val:.5f} ({A_norms.avg:.5f})  '
            'Relu_b {Relu_bs.val:.5f} ({Relu_bs.avg:.5f})  '
            'All {all_val:.5f} ({all_avg:.5f})  \n'.format(
            t, eps, batch_time=batch_time,
            loss=losses, errors=errors, robust_errors = robust_errors, l1_loss = l1_losses,
            regular_ce_loss = regular_ce_losses, robust_ce_loss = robust_ce_losses, 
            unstable = unstable_neurons, dead = dead_neurons, alive = alive_neurons,
            tight = relu_activities, bias = bound_bias, diff = bound_diff,
            model_range = model_range, 
            kappa = kappa, beta=beta,Axs = A_xs, not_relu_b = not_ReLU_lower_bs, A_norms = A_norms, Relu_bs = ReLU_lower_bs , all_val=all_val, all_avg =all_avg
    ))

        for i, l in enumerate(model if isinstance(model, BoundSequential) else model.module):
            if isinstance(l, BoundLinear) or isinstance(l, BoundConv2d):
                norm = l.weight.data.detach().view(l.weight.size(0), -1).abs().sum(1).max().cpu()
                logger.log('layer {} norm {}'.format(i, norm))
            
    if model.bound_opts['ours'] and train is False :
        torch.set_grad_enabled(False)
     
    if method == "natural":
        return errors.avg, errors.avg
    else:
        return robust_errors.avg, errors.avg

    
def Train_calloss_ours(loss,robust_ce,model, t, i, data, labels, loader, eps,end_eps, max_eps, norm, logger, verbose, train, method, cal_grad = False,beta=1.0, kappa=0.0,cal_lb=False,cal_loss=False, frozen =5, multistep=False,**kwargs):

    if multistep:
        niters=7
        lower_d_list2=model.lower_d_list
        for nn in range (niters):
            if nn>=1:
                loss, _ = Train_calloss(model, t, i, data, labels, loader, eps,end_eps, max_eps, norm, logger, verbose, train, method, lower_d_list2=lower_d_list2, beta=beta, kappa=kappa,cal_grad=True,**kwargs)
            grad = torch.autograd.grad(loss,lower_d_list2,retain_graph=False, create_graph=False)
            for list_i,g in enumerate(grad):
                eta = g.sign()*0.1 
                lower_d2 = torch.clamp(lower_d_list2[list_i]-eta, min=0, max=1)

                if (nn == niters-1):
                    lower_d2=lower_d2.detach()

                lower_d_list2[list_i]=lower_d2
                del g, eta

    else:
        lower_d_list2=[]
        grad = torch.autograd.grad(loss, model.lower_d_list,retain_graph=False, create_graph=False)
        for list_i,g in enumerate(grad):
            eta = g.sign()*2  
            lower_d2 = torch.clamp(model.lower_d_list[list_i]-eta, min=0, max=1).detach()
            lower_d_list2.append(lower_d2.type(torch.cuda.FloatTensor))
            del g, eta

    del  robust_ce, grad, loss
    loss, robust_ce, clb, bias = Train_calloss(model, t, i, data, labels, loader, eps,end_eps, max_eps, norm, logger, verbose, train, method, lower_d_list2=lower_d_list2, beta=beta, kappa=kappa,cal_grad=True,cal_lb=True,frozen=frozen, **kwargs)
    del  model.lower_d_list
    
    return loss, robust_ce, clb, bias
                

    
    
def main(args):
    config = load_config(args)
    global_train_config = config["training_params"]
    models, model_names = config_modelloader(config) 
    for model, model_id, model_config in zip(models, model_names, config["models"]):
        train_config = copy.deepcopy(global_train_config)
        
        if "training_params" in model_config:
            train_config = update_dict(train_config, model_config["training_params"])
        model = BoundSequential.convert(model, train_config["method_params"]["bound_opts"])
        model.define_bound_opts(train_config["method_params"]["bound_opts"])
        model_id = model_id +'_'+ train_config["method_params"]["bound_type"] +'_'+ str(train_config["train_epsilon"]) +'_'+ train_config["name"]
        epochs = train_config["epochs"]
        lr = train_config["lr"]
        weight_decay = train_config["weight_decay"]
        starting_epsilon = train_config["starting_epsilon"]
        end_epsilon = train_config["epsilon"]
        train_end_epsilon = train_config["train_epsilon"]

        schedule_length = train_config["schedule_length"]
        schedule_start = train_config["schedule_start"]
        optimizer = train_config["optimizer"]
        method = train_config["method"]
        verbose = train_config["verbose"]
        lr_decay_step = train_config["lr_decay_step"]
        lr_decay_milestones = train_config["lr_decay_milestones"]
        lr_decay_factor = train_config["lr_decay_factor"]
        multi_gpu = train_config["multi_gpu"]
        method_param = train_config["method_params"]
        norm = float(train_config["norm"])
        train_data, test_data = config_dataloader(config, **train_config["loader_params"])
        
        n_loss = train_config["n_loss"]
        cal_loss = train_config["cal_loss"]
        max_loss = train_config["max_loss"]
        min_loss = train_config["min_loss"]
        
        cal_grad = train_config["cal_grad"]
        cal_grad_norm = train_config["cal_grad_norm"]
        start_beta = train_config["start_beta"]
        start_kappa = train_config["start_kappa"]
        lr_consist = train_config["lr_consist"]
        
        frozen_dict = train_config["frozen_dict"]
        save_model = train_config["save"]        
        
        normalize=train_config["loader_params"]["normalize_input"]
        
        bound_eval=train_config["bound_eval"]
        if bound_eval:
            bound_opt_evals=[]
            bound_opt_evals.append({'same-slope': False, 'zero-lb': False, 'one-lb': False, 'binary': False, 'ours': True, 'uniform': False}) ##ours               
            bound_opt_evals.append({'same-slope': False, 'zero-lb': False, 'one-lb': False, 'binary': False, 'ours': False, 'uniform': False}) ##crown ibp
            bound_opt_evals.append({'same-slope': True, 'zero-lb': False, 'one-lb': False, 'binary': False, 'ours': False, 'uniform': False}) ## cap
            bound_opt_evals.append({'same-slope': False, 'zero-lb': False, 'one-lb': False, 'binary': True, 'ours': False, 'uniform': False}) ##binary     
            bound_opt_evals.append({'same-slope': False, 'zero-lb': False, 'one-lb': False, 'binary': False, 'ours': False, 'uniform': True}) ##uniform        
            if train_config["method_params"]["bound_opts"]["ours"]:
                bound_opt_evals.remove({'same-slope': False, 'zero-lb': False, 'one-lb': False, 'binary': False, 'ours': True, 'uniform': False}) ##ours               
            elif train_config["method_params"]["bound_opts"]["same-slope"]:
                bound_opt_evals.remove({'same-slope': True, 'zero-lb': False, 'one-lb': False, 'binary': False, 'ours': False, 'uniform': False}) ## cap
            elif train_config["method_params"]["bound_opts"]["binary"]:
                bound_opt_evals.remove({'same-slope': False, 'zero-lb': False, 'one-lb': False, 'binary': True, 'ours': False, 'uniform': False}) ##binary
            elif train_config["method_params"]["bound_opts"]["uniform"]:
                bound_opt_evals.remove({'same-slope': False, 'zero-lb': False, 'one-lb': False, 'binary': False, 'ours': False, 'uniform': True}) ##uniform   
            else:     
                bound_opt_evals.remove({'same-slope': False, 'zero-lb': False, 'one-lb': False, 'binary': False, 'ours': False, 'uniform': False}) ##crown ibp

        if optimizer == "adam":
            opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == "sgd":
            opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
        else:
            raise ValueError("Unknown optimizer")
       
        batch_multiplier = train_config["method_params"].get("batch_multiplier", 1)
        batch_size = train_data.batch_size * batch_multiplier  
        num_steps_per_epoch = int(np.ceil(1.0 * len(train_data.dataset) / batch_size))
        print('num_steps_per_epoch',num_steps_per_epoch)
        epsilon_scheduler = EpsilonScheduler(train_config.get("schedule_type", "linear"), schedule_start * num_steps_per_epoch, ((schedule_start + schedule_length) - 1) * num_steps_per_epoch, starting_epsilon, end_epsilon, num_steps_per_epoch)
        max_eps = end_epsilon
        
        train_epsilon_scheduler = EpsilonScheduler(train_config.get("schedule_type", "linear"), schedule_start * num_steps_per_epoch, ((schedule_start + schedule_length) - 1) * num_steps_per_epoch, starting_epsilon, train_end_epsilon, num_steps_per_epoch) ##0818 JS
        train_max_eps = train_end_epsilon 
        
        if lr_decay_step:
            # Use StepLR. Decay by lr_decay_factor every lr_decay_step.
            lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=lr_decay_step, gamma=lr_decay_factor)
            lr_decay_milestones = None
        elif lr_decay_milestones:
            # Decay learning rate by lr_decay_factor at a few milestones.
            lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=lr_decay_milestones, gamma=lr_decay_factor)
        else:
            raise ValueError("one of lr_decay_step and lr_decay_milestones must be not empty.")
            
        model_name = get_path(config, model_id, "model", load = False)
        best_model_name = get_path(config, model_id, "best_model", load = False)      
        model_log = get_path(config, model_id, "train_log")
        loss_log = model_log+'loss'
        grad_log = model_log+'grad'
        grad_norm_log = model_log+'grad_norm'
        a_sign_log = model_log+'a_sign'
        cosine = model_log+'cosine'
        loss_max = model_log+'loss_max'
        logger = Logger(open(model_log, "w"),open(loss_log, "w"),open(grad_log, "w"),open(grad_norm_log, "w"),open(a_sign_log, "w"),open(cosine, "w"),open(loss_max,'w'))
        logger.log(model_name)
        logger.log("Command line:", " ".join(sys.argv[:]))
        logger.log("training configurations:", train_config)
        logger.log("Model structure:")
        logger.log(str(model))
        logger.log("data std:", train_data.std)
        best_err = np.inf
        recorded_pgd_err=np.inf
        recorded_clean_err = np.inf
        timer = 0.0
         
        if multi_gpu:
            logger.log("\nUsing multiple GPUs for computing CROWN-IBP bounds\n")
            model = BoundDataParallel(model) 
        model = model.cuda()

        frozen = 13 
        for t in range(epochs):  
            train_epoch_start_eps = train_epsilon_scheduler.get_eps(t, 0) 
            train_epoch_end_eps = train_epsilon_scheduler.get_eps(t+1, 0)
            epoch_start_eps = epsilon_scheduler.get_eps(t, 0)
            epoch_end_eps = epsilon_scheduler.get_eps(t+1, 0)            
            
            logger.log("Epoch {}, learning rate {}, epsilon {:.6g} - {:.6g}".format(t, lr_scheduler.get_lr(), train_epoch_start_eps, train_epoch_end_eps))          
            start_time = time.time() 
            
            for ff, (frozen_t,frozen_value) in enumerate(zip(frozen_dict.keys(),frozen_dict.values())):
                if t >= int(frozen_t):
                    frozen = frozen_value
            if train_config["method_params"]["bound_type"] == "crown-interval-frozen":
                logger.log("Frozen value {}".format(frozen))  
            
            Train(model,t, train_data, train_epsilon_scheduler, train_max_eps, norm, logger, verbose, True, opt, method, cal_loss=cal_loss, n_loss=n_loss,max_loss=max_loss, min_loss=min_loss, cal_grad = cal_grad, config = config,cal_grad_norm=cal_grad_norm, start_beta=start_beta,start_kappa=start_kappa,lr_consist=lr_consist, frozen=frozen,  **method_param)
                
            if lr_consist ==False:
                if lr_decay_step:
                    lr_scheduler.step(epoch=max(t - (schedule_start + schedule_length - 1) + 1, 0))
                elif lr_decay_milestones:
                    lr_scheduler.step()
            epoch_time = time.time() - start_time
            timer += epoch_time
            logger.log('Epoch time: {:.4f}, Total time: {:.4f}'.format(epoch_time, timer))
            logger.log("Evaluating...")
            with torch.no_grad():                
                if bound_eval and t >= (schedule_start):
                    for _,bound_opt_eval in enumerate(bound_opt_evals):
                        model.convert_bounds(bound_opt_eval)
                        Train(model,t, test_data, EpsilonScheduler("linear", 0, 0, epoch_end_eps, epoch_end_eps, 1), max_eps, norm, logger, verbose, False, None, method, start_beta=start_beta,start_kappa=start_kappa,bound_opt_eval=bound_opt_eval,frozen=frozen,  **method_param)
                    model.convert_bounds(train_config["method_params"]['bound_opts'])
               
                err, clean_err = Train(model,t, test_data, EpsilonScheduler("linear", 0, 0, epoch_end_eps, epoch_end_eps, 1), max_eps, norm, logger, verbose, False, None, method, start_beta=start_beta,start_kappa=start_kappa,frozen=frozen, **method_param)
                    
      
            logger.log('saving to', model_name)
            torch.save({
                    'state_dict' : model.module.state_dict() if multi_gpu else model.state_dict(), 
                    'epoch' : t,
                    }, model_name)
            # save the best model after we reached the schedule
            if t >= (schedule_start + schedule_length):
                if err <= best_err:
                    best_err = err
                    recorded_clean_err = clean_err
                    
                    if normalize:
                        pgd_err = evaluate_pgd_n(loader=test_data,model=model,norm=norm, epsilon=max_eps, alpha=max_eps/4, niters=100)
                    else:
                        pgd_err = evaluate_pgd(loader=test_data,model=model,norm=norm, epsilon=max_eps, alpha=max_eps/4, niters=100)       
                    recorded_pgd_err = pgd_err
                    if(recorded_pgd_err>best_err):
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    
                    logger.log('Saving best model {} with error {}, pgd {}, clean {}'.format(best_model_name, best_err,recorded_pgd_err,recorded_clean_err))
                    torch.save({
                            'state_dict' : model.module.state_dict() if multi_gpu else model.state_dict(), 
                            'robust_err' : err,
                            'pgd_err' : pgd_err,
                            'clean_err' : clean_err,
                            'epoch' : t,
                            }, best_model_name)
            if save_model:
                if t%save_model ==0:
                    torch.save({
                            'state_dict' : model.module.state_dict() if multi_gpu else model.state_dict(), 
                            'epoch' : t,
                            }, model_name+str(t))

        logger.log('Total Time: {:.4f}'.format(timer))
        logger.log('Model {} best err {}, pgd_err {:.4f}, clean err {}'.format(model_id, best_err, recorded_pgd_err,recorded_clean_err))
        logger.log('{:.4f}/{:.4f}/{:.4f}'.format(recorded_clean_err, recorded_pgd_err, best_err))


if __name__ == "__main__":
    args = argparser()
    print('torch version: ',torch.__version__)
    main(args)
