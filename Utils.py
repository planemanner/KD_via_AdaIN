# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Function
from os import path
import shutil

def calc_mean_std(features, eps):

    """
    :param features: shape of features -> [batch_size, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
    """
    batch_size, c = features.size()[:2]

    features_mean = features.view(batch_size, c, -1).mean(dim=2).view(batch_size, c, 1, 1)

    features_std = features.view(batch_size, c, -1).std(dim=2).view(batch_size, c, 1, 1) + eps

    return features_mean, features_std

"""
Statistical Matching Loss

"""
def SM_Loss(T_maps, S_maps, eps=1e-7):
    
    T_mean, T_std = calc_mean_std(T_maps, eps)
    S_mean, S_std = calc_mean_std(S_maps, eps)
    
    mean_match = torch.pow((T_mean - S_mean), 2)
    std_match = torch.pow((T_std - S_std), 2)
    
    L_SM = (mean_match+std_match).mean(dim=1)
    
    return torch.sum(L_SM)


"""
Adaptive Instance Normalization
 
"""
def AdaIN(content_features, style_features):

    """
    Adaptive Instance Normalization
    :param content_features: shape -> [batch_size, c, h, w]
    :param style_features: shape -> [batch_size, c, h, w]
    :return: normalized_features shape -> [batch_size, c, h, w]
    """
    eps = 1e-7
    
    content_mean, content_std = calc_mean_std(content_features, eps)

    style_mean, style_std = calc_mean_std(style_features, eps)

    normalized_features = style_std * (content_features - content_mean) / content_std + style_mean

    return normalized_features

def bn_statistics(T_block, S_block):
    
    T_bn = T_block.layer[-1].bn2
    S_bn = S_block.layer[-1].bn2
    
    T_gamma = T_bn.weight.data
    T_beta = T_bn.bias.data
    
    S_gamma = S_bn.weight.data
    S_beta = S_bn.bias.data
    
    mean_T_gamma, std_T_gamma = bn_mean_std(T_gamma)
    mean_S_gamma, std_S_gamma = bn_mean_std(S_gamma)
    
    mean_T_beta, std_T_beta = bn_mean_std(T_beta)
    mean_S_beta, std_S_beta = bn_mean_std(S_beta)
    
    """
    batch normalization matching 
    """
    
    const_gamma = (mean_T_gamma - mean_S_gamma) + (std_T_gamma-std_S_gamma)
    const_beta = (mean_T_beta - mean_S_beta) + (std_T_beta-std_S_beta)
    
    return const_beta + const_gamma
    

def bn_mean_std(weight):
    
    mean_bn_weight = torch.mean(weight)
    std_bn_weight = torch.std(weight)
    
    return mean_bn_weight, std_bn_weight
    


def accuracy(output, target, topk=(1,)):
    if len(target.shape)>1:
        target = torch.argmax(target, dim=1)
    
    maxk = max(topk)
    batch_size = target.size(0)
    """
    top k sorting
    """
    _, pred = output.topk(maxk, 1, True, True) 
    pred = pred.t()
    correct = pred.eq(target.view(1,-1).expand_as(pred))
    res = []
    
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100/batch_size)) 
    
    return res

class AverageMeter():
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum/self.count
        
def evaluator(testloader, model):

    top1 = AverageMeter()
    top5 = AverageMeter()
    
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1,5))
            
            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))
    
    model.train()
    return top1.avg, top5.avg
            
def save_ckpt(state, is_best, root_path = 'checkpoint', file_name = 'checkpoint.pth.tar'):
    file_path = path.join(root_path, file_name)
    torch.save(state, file_path)
    if is_best:
        shutil.copyfile(file_path, path.join(root_path, 'model_best.pth.tar'))    