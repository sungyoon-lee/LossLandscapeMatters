
import sys
import copy
import torch
import numpy as np
from torch.nn import Sequential, Linear, ReLU, CrossEntropyLoss

from bound_layers import BoundSequential, BoundLinear, BoundConv2d, BoundDataParallel
# from gpu_profile import gpu_profile
import time
from datetime import datetime
from eps_scheduler import EpsilonScheduler
from config import load_config, get_path, config_modelloader, config_dataloader
from argparser import argparser
from train import AverageMeter, Logger
# sys.settrace(gpu_profile)
from pgd_eval import evaluate_pgd, evaluate_pgd_n

import seaborn as sns
import matplotlib.pyplot as plt
    
DEBUGG = False
BREAK = False


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
        #logger.log('eps {} close to 0, using natural training'.format(end_eps))
        method = "natural"
##########################    for i, (data, labels) in enumerate(loader): 
    start = time.time()
    #eps = eps_scheduler.get_eps(t, int(i//batch_multiplier)) 
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
            #######0730
            #print('cal_loss',norm, data_ub[0][0], data_lb[0][0], eps, labels[:3])
            
            ub, ilb, relu_activity, unstable, dead, alive = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, method_opt="interval_range")
            if beta < 1e-5:
                lb = ilb
            else:
                if kwargs["runnerup_only"]:
                    # regenerate a smaller c, with just the runner-up prediction
                    # mask ground truthlabel output, select the second largest class
                    # print(output)
                    # torch.set_printoptions(threshold=5000)
                    masked_output = output.detach().scatter(1, labels.unsqueeze(-1), -100)
                    # print(masked_output)
                    # location of the runner up prediction
                    runner_up = masked_output.max(1)[1]
                    # print(runner_up)
                    # print(labels)
                    # get margin from the groud-truth to runner-up only
                    runnerup_c = torch.eye(num_class).type_as(data)[labels]
                    # print(runnerup_c)
                    # set the runner up location to -
                    runnerup_c.scatter_(1, runner_up.unsqueeze(-1), -1)
                    runnerup_c = runnerup_c.unsqueeze(1).detach()
                    # print(runnerup_c)
                    # get the bound for runnerup_c
                    _, _, clb, bias = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, method_opt="backward_range")
                    clb = clb.expand(clb.size(0), num_class - 1)
                else:
                    # get the CROWN bound using interval bounds 
                    _, _, clb, bias = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, method_opt="backward_range",lower_d_list2=lower_d_list2)
                # how much better is crown-ibp better than ibp?
                diff = (clb - ilb).sum().item()
                # lb = torch.max(lb, clb)
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
 #               print(lb.shape)                
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
    
    All_list=[]
    loss_list=[]
    
    

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
        start = time.time()
        eps = eps_scheduler.get_eps(t, int(i//batch_multiplier)) 
        crown_final_beta = kwargs['final-beta'] #
        natural_final_factor = kwargs["final-kappa"] #
        beta = start_beta - (1.0-(max_eps-eps)/max_eps)*(start_beta-crown_final_beta)  #
        kappa = start_kappa - (1.0-(max_eps-eps)/max_eps)*(start_kappa-natural_final_factor) #
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

        '''
        torch.set_printoptions(threshold=5000)
        print('prediction:  ', output)
        ub, lb, _, _, _, _ = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, method_opt="interval_range")
        lb = lb_s.scatter(1, sa_labels, lb)
        ub = ub_s.scatter(1, sa_labels, ub)
        print('interval ub: ', ub)
        print('interval lb: ', lb)
        ub, _, lb, _ = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, upper=True, lower=True, method_opt="backward_range")
        lb = lb_s.scatter(1, sa_labels, lb)
        ub = ub_s.scatter(1, sa_labels, ub)
        print('crown-ibp ub: ', ub)
        print('crown-ibp lb: ', lb) 
        ub, _, lb, _ = model(norm=norm, x_U=data_ub, x_L=data_lb, eps=eps, C=c, upper=True, lower=True, method_opt="full_backward_range")
        lb = lb_s.scatter(1, sa_labels, lb)
        ub = ub_s.scatter(1, sa_labels, ub)
        print('full-crown ub: ', ub)
        print('full-crown lb: ', lb)
        input()
        '''

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
                        # print(output)
                        # torch.set_printoptions(threshold=5000)
                        masked_output = output.detach().scatter(1, labels.unsqueeze(-1), -100)
                        # print(masked_output)
                        # location of the runner up prediction
                        runner_up = masked_output.max(1)[1]
                        # print(runner_up)
                        # print(labels)
                        # get margin from the groud-truth to runner-up only
                        runnerup_c = torch.eye(num_class).type_as(data)[labels]
                        # print(runnerup_c)
                        # set the runner up location to -
                        runnerup_c.scatter_(1, runner_up.unsqueeze(-1), -1)
                        runnerup_c = runnerup_c.unsqueeze(1).detach()
                        # print(runnerup_c)
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
                    # get the CROWN bound using interval bounds 
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
            
            All=(model.ReLU_lower_b.sum()+model.A_x.sum()+model.not_ReLU_lower_b.sum()+model.A_norm.sum())/data.size(0)
            All_list.append(All.item())
            loss_list.append(loss.item())

            
        elif (verbose or method != "natural") and kwargs["bound_type"] == "crown-interval-frozen" and beta >= 1e-5 and not kwargs["runnerup_only"]:
            diff = (clb - ilb).sum().item()
            lb = clb * beta + ilb * (1 - beta)
    
        if train: 
            opt.zero_grad()
            loss.backward()

            if cal_loss and (t*len(loader)+i)%100 ==0:
                model_list = [qq for qq in model.state_dict().keys()]
                dict_all_tmp = {}
                lr = opt.state_dict()['param_groups'][0]['lr']
                original_grad = []
                for point in loss_points:
                    dict_all_tmp[point]={}

                for v, param in enumerate(model.parameters()):
                    param_grad = param.grad.data.detach()
                    original_grad.append(param_grad.view(1,-1)) ######flatten
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
                        loss_tmp, Rloss_tmp, clb, bias=Train_calloss_ours(loss_tmp,Rloss_tmp,model_loss, t, i, data, labels, loader, eps,end_eps, max_eps, norm, logger, verbose, train, method, beta=beta, kappa=kappa,cal_grad=True,cal_lb=True,multistep=False,frozen=frozen, **kwargs) # if you want to test multistep, then set multistep=True
                        
                        
                
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
            # robust_ce_losses.update(robust_ce, data.size(0))
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
        return errors.avg, errors.avg,loss_list,All_list
    else:
        return robust_errors.avg, errors.avg ,loss_list, All_list
    
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
            lower_d_list2.append(lower_d2.type(torch.cuda.LongTensor))
            del g, eta

    del  robust_ce, grad, loss
    loss, robust_ce, clb, bias = Train_calloss(model, t, i, data, labels, loader, eps,end_eps, max_eps, norm, logger, verbose, train, method, lower_d_list2=lower_d_list2, beta=beta, kappa=kappa,cal_grad=True,cal_lb=True,frozen=frozen, **kwargs)
    del  model.lower_d_list

    
    return loss, robust_ce, clb, bias
    
def main(args):
    config = load_config(args)
    global_eval_config = config["eval_params"]
    config["models"][0]["model_id"]=config["eval_params"]["model_paths"]
    models, model_names = config_modelloader(config, load_pretrain = True)


    robust_errs = []
    pgd_errs = []
    errs = []
    t=1000000
    
    losses_list=[]
    Alls_list=[]
    for model, model_id, model_config in zip(models, model_names, config["models"]):
        eval_config = copy.deepcopy(global_eval_config)
        if "eval_params" in model_config:
            eval_config.update(model_config["eval_params"])
        
        model = BoundSequential.convert(model, eval_config["method_params"]["bound_opts"])
        model.define_bound_opts(eval_config["method_params"]["bound_opts"])

        normalize= eval_config["loader_params"]["normalize_input"]
        
        bound_eval=eval_config["bound_eval"] ##bound eval setting is for evaluate a model with other bound methods (eval_params:bound_eval=true)
        if bound_eval:
            bound_opt_evals=[]
            bound_opt_evals.append({'same-slope': False, 'zero-lb': False, 'one-lb': False, 'binary': False, 'ours': True, 'uniform': False}) ##ours               
            bound_opt_evals.append({'same-slope': False, 'zero-lb': False, 'one-lb': False, 'binary': False, 'ours': False, 'uniform': False}) ##crown ibp
            bound_opt_evals.append({'same-slope': True, 'zero-lb': False, 'one-lb': False, 'binary': False, 'ours': False, 'uniform': False}) ## cap
            bound_opt_evals.append({'same-slope': False, 'zero-lb': False, 'one-lb': False, 'binary': True, 'ours': False, 'uniform': False}) ##binary     
            bound_opt_evals.append({'same-slope': False, 'zero-lb': False, 'one-lb': False, 'binary': False, 'ours': False, 'uniform': True}) ##uniform        
            if eval_config["method_params"]["bound_opts"]["ours"]:
                bound_opt_evals.remove({'same-slope': False, 'zero-lb': False, 'one-lb': False, 'binary': False, 'ours': True, 'uniform': False}) ##ours               
            elif eval_config["method_params"]["bound_opts"]["same-slope"]:
                bound_opt_evals.remove({'same-slope': True, 'zero-lb': False, 'one-lb': False, 'binary': False, 'ours': False, 'uniform': False}) ## cap
            elif eval_config["method_params"]["bound_opts"]["binary"]:
                bound_opt_evals.remove({'same-slope': False, 'zero-lb': False, 'one-lb': False, 'binary': True, 'ours': False, 'uniform': False}) ##binary
            elif eval_config["method_params"]["bound_opts"]["uniform"]:
                bound_opt_evals.remove({'same-slope': False, 'zero-lb': False, 'one-lb': False, 'binary': False, 'ours': False, 'uniform': True}) ##uniform   
            else:     
                bound_opt_evals.remove({'same-slope': False, 'zero-lb': False, 'one-lb': False, 'binary': False, 'ours': False, 'uniform': False}) ##crown ibp


        
        model = model.cuda()
        # read training parameters from config file
        method = eval_config["method"]
        verbose = eval_config["verbose"]
        eps = eval_config["epsilon"]
        # parameters specific to a training method
        method_param = eval_config["method_params"]
        norm = float(eval_config["norm"])
        train_data, test_data = config_dataloader(config, **eval_config["loader_params"])

        model_name = get_path(config, model_id, "model", load = False)
        print(model_name)
        model_log = get_path(config, model_id, "eval_log")
        logger = Logger(open(model_log, "w"))

        logger.log("evaluation configurations:", eval_config)
            
        logger.log("Evaluating...")
        print("Model:",model)
        
        with torch.no_grad():
            # evaluate
                    
            if bound_eval:
                for _,bound_opt_eval in enumerate(bound_opt_evals):
                    model.convert_bounds(bound_opt_eval)
                    _,_,loss_list,All_list=Train(model, t, test_data, EpsilonScheduler("linear", 0, 0, eps, eps, 1), eps, norm, logger, verbose, False, None, method, bound_opt_eval=bound_opt_eval, **method_param)
                    losses_list.append(loss_list)
                    Alls_list.append(All_list)
                model.convert_bounds(eval_config["method_params"]['bound_opts'])

            robust_err, err, loss_list, All_list = Train(model, t, test_data, EpsilonScheduler("linear", 0, 0, eps, eps, 1), eps, norm, logger, verbose, False, None, method,  **method_param)
            losses_list.append(loss_list)
            Alls_list.append(All_list)
            
        
        if normalize:
            pgd_err = evaluate_pgd_n(loader=test_data,model=model,norm=norm, epsilon=eps, alpha=eps/4, niters=100)
        else:
            pgd_err = evaluate_pgd(loader=test_data,model=model,norm=norm, epsilon=eps, alpha=eps/4, niters=100)       
        robust_errs.append(robust_err)
        pgd_errs.append(float(pgd_err.item()))
        errs.append(err)
            

    robust_errs = np.array(robust_errs)
    i_min = np.argmin(robust_errs)
    i_max = np.argmax(robust_errs)
    i_median = np.argsort(robust_errs)[len(robust_errs) // 2]
  
 
    print("-------------------------------------")
    print('{:.4f}/{:.4f}/{:.4f}'.format(np.mean(errs),np.mean(pgd_errs),np.mean(robust_errs)))
    
    if global_eval_config["violin_plot"]: ## if you want to get loss and tightness (set eval_params:bound_eval=true and eval_params:violin_plot=true)
        print("-------------------------------------")
        print("LOSS=",losses_list)
        print("ALL=",Alls_list)


if __name__ == "__main__":
    args = argparser()
    main(args)
